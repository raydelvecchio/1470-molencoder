import tensorflow as tf
from preprocess import DICT_NAME, TRAIN_NAME, TEST_NAME, OUT_FILE, MAX_SMILE_LENGTH, one_hot_smile, pad_smile
import h5py
from model import Model
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 1000


def train_model(model, data, batch_size=BATCH_SIZE):
    switch = True
    optimizer_max = tf.keras.optimizers.Adam(learning_rate=0.01)
    optimizer_min = tf.keras.optimizers.Adam(learning_rate=0.001)
    total_loss = 0
    loss_list = list()
    batch_list = list()
    for i in range(0, data.shape[0], batch_size):  # loop over all training examples we have
        inputs = data[i:i+batch_size]  # creating a batch of inputs here
        with tf.GradientTape() as tape:
            out, mu, logvar = model.call(inputs)
            loss = model.loss(out, inputs, mu, logvar)
            print("Batch " + str(i / 1000) + " loss: " + str(float(loss)))
            loss_list.append(loss)
            batch_list.append(i/1000)
            total_loss += loss
        gradient = tape.gradient(loss, model.trainable_variables)
        if i % 10000 == 0 and i != 0:  # switch LR every 10k samples; "cyclical learning rate" to avoid local min
            switch = not switch
            print("Changing learning rate...")
        if switch:
            optimizer_max.apply_gradients(zip(gradient, model.trainable_variables))
        else:
            optimizer_min.apply_gradients(zip(gradient, model.trainable_variables))

    plt.plot(batch_list, loss_list, color='black', linewidth=3.5)
    plt.title('Loss by Batch', fontsize=14)
    plt.xlabel('Batch Number')
    plt.ylabel('VAE Loss')
    plt.gcf().set_size_inches(10, 5)
    plt.show()

    return total_loss


def generate_molecules(model, character_dict, smile):  # this acts as our test function, as specified in devpost
    """
    Takes in a smile string, then outputs a molecule similar to it by sampling from a learned distribution.
    :param model: TRAINED model, pretty self-explanatory
    :param character_dict: dictionary of character in training set
    :param smile: smile string that we want to use as a base molecule
    :return: smile string of similar molecule generated by our trained model
    """
    one_hot = one_hot_smile(pad_smile(smile), character_dict, preprocess=False)
    one_reshape = np.repeat(one_hot, BATCH_SIZE)  # need this to be compatible with linear layers
    reshape = tf.reshape(one_reshape, [BATCH_SIZE, MAX_SMILE_LENGTH, 55])
    output, _, _ = model.call(reshape)  # select the first output of linear layers; they're all the same
    distribution = output[0]

    new_smile = ""
    for i in range(len(distribution)):
        target = distribution[i]  # gets appropriate distribution amongst characters; this should sum to one!
        probabilities = create_relative_probabilities(target)
        sampled_char_idx = np.random.choice(np.arange(len(character_dict)), p=probabilities)  # samples from dist
        new_smile += character_dict[sampled_char_idx].decode('utf-8')
    return new_smile


def create_relative_probabilities(char_dist):
    """
    Bootleg fix to softmax running over the wrong thing in the decoder. Given some data, it normalizes it to it's
    all proportional to the original data, but sums up to one (for input into random sampling in generate
    molecule). Does this by finding the total sum, then creating a list where each value is = value / total, or its
    % capitalization on the total data. Ultimate goal is to fix the "does not sum to 1" error in np.random.choice.
    NOTE: last probability is always that of the " " (space) character.
    :param char_dist: (smile_length, dict_lengh) output from the model with data on a character distribution
    :return: list of probabilities of each character
    """
    total = np.sum(char_dist)
    proportion_list = list()
    for value in char_dist:
        proportion = float(value / total)
        proportion_list.append(round(proportion, 3))  # rounds proportion to 4 decimals; make convergence to 1 easier

    # BEGIN SUM TO 1 CORRECTION HERE
    difference = 1 - np.sum(proportion_list[:-1])  # finds sum of all values but last one
    if difference >= 0:  # if different is positive, last value is equal to difference
        proportion_list[-1] = difference
    else:  # cannot have negative numbers in probability list, so we add it to something that can absorb it
        assigned = False
        for i in range(len(proportion_list)):
            if proportion_list[i] > abs(difference) and not assigned:
                proportion_list[i] += difference
                assigned = True
    # END SUM TO 1 CORRECTION HERE

    return proportion_list


def interpret_result(smile_string):
    """
    Changes smile string to the first consecutive sequence of characters, hence removing unnecessary spaces.
    :param smile_string: smile string to post-process
    :return: smile string without all spaces at the end
    """
    counter = 0
    for char in smile_string:
        if char == " ":
            return smile_string[:counter]
        counter += 1
    return smile_string


def main():
    """
    Reads data from chembl22/chembl22.h5, trains model, then tests model!
    :return:
    """
    data = h5py.File(OUT_FILE)
    train = data[TRAIN_NAME][:]
    test = data[TEST_NAME][:]
    char_dict = list(data[DICT_NAME][:])

    print("Making model...")
    molencoder = Model()

    print("Training...")
    total_loss = train_model(molencoder, train)

    print("Generating similar molecule...")
    new_mol = generate_molecules(molencoder, char_dict, "HC(H)=C(H)(H)")
    new_mol = interpret_result(new_mol)
    print("New Molecule: " + new_mol)
    new_mol = generate_molecules(molencoder, char_dict, "CCC")
    new_mol = interpret_result(new_mol)
    print("New Molecule: " + new_mol)
    new_mol = generate_molecules(molencoder, char_dict, "CH")
    new_mol = interpret_result(new_mol)
    print("New Molecule: " + new_mol)


if __name__ == "__main__":
    main()