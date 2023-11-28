import pandas as pd
from sklearn.model_selection import train_test_split as split
import numpy as np
import h5py

# hyperparameters for dataset processing
MAX_SMILE_LENGTH = 120
SAMPLE_NUM = 500_000  # we have a total of 1,678,393 molecules available; preprocess time depends on this; 500k base
SMILES_COL_NAME = 'canonical_smiles'  # this is default column name
TEST_SIZE = 0.2
IN_FILE = 'data/chembl_22_chemreps.txt.gz'
OUT_FILE = 'chembl22/chembl22.h5'
DICT_NAME = 'dictionary'
TRAIN_NAME = 'train_encodings'
TEST_NAME = 'test_encodings'


def preprocess():
    """
    dataset from: ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz.
    Reads compressed molecule data and preprocesses it. Saves output to h5py file for data storage!
    :returns:
    train_indices: (len(train), MAX_SMILE_LENGTH, len(char_dict)) shaped numpy array representing one-hot encodings for all letters for all train SMILEs
    test_indices: (len(test), MAX_SMILE_LENGTH, len(char_dict)) shaped numpy array representing one-hot encodings for all letters for all test SMILEs
    character_dict: list of all characters present in our dataset
    """
    print("Accessing data...")
    chembl22 = pd.read_table(IN_FILE, compression='gzip')
    within_length = chembl22[SMILES_COL_NAME].map(len) <= MAX_SMILE_LENGTH  # list of true/false if under length
    chembl22 = chembl22[within_length].sample(n=SAMPLE_NUM)
    smiles_strings = chembl22[SMILES_COL_NAME]

    character_dict = create_chardict(smiles_strings)
    character_dict.append(' ')  # appends the padding character whitespace; this is always the last character

    print("Processing data... this may take a while...")
    train_smiles, test_smiles = split(smiles_strings, shuffle=True, test_size=TEST_SIZE)
    train_smiles = np.array(train_smiles.map(lambda x: one_hot_smile(x, character_dict, preprocess=True)))
    test_smiles = np.array(test_smiles.map(lambda x: one_hot_smile(x, character_dict, preprocess=True)))

    print("Loading data into h5py file...")
    with h5py.File(OUT_FILE, 'w') as f:  # saves all data to h5 file; copies over train_smiles and test_smiles to data
        f.create_dataset(DICT_NAME, data=[char.encode('utf-8') for char in character_dict])

        train = f.create_dataset(TRAIN_NAME, shape=(len(train_smiles), MAX_SMILE_LENGTH, len(character_dict)))
        for i in range(len(train)):
            train[i] = train_smiles[i]

        test = f.create_dataset(TEST_NAME, shape=(len(test_smiles), MAX_SMILE_LENGTH, len(character_dict)))
        for i in range(len(test)):
            test[i] = test_smiles[i]

    print("Preprocess complete!")
    return train_smiles, test_smiles, character_dict


def create_chardict(strings):
    """
    Given a list of strings, creates a dictionary list of all characters in all strings.
    :param strings: list, array, or dataframe of strings to find dictionary
    :return: list of all characters in strings
    """
    charset = set()
    for string in strings:
        for char in string:
            charset.add(char)
    charlist = list(charset)
    return charlist


def one_hot_smile(smile_string, character_dict, preprocess):
    """
    Turns a smile string into a one-hot encoded vector for each letter in dictionary.
    :param smile_string: string to turn into one-hot vectors
    :param character_dict: list dictionary containing all letters
    :return: numpy array representation of the vector, containing a one-hot array over dictionary
    for each letter in smile_string. (EX: if first letter is 'c', the first row is a one-hot array with [0, 0, 1, 0],
    where 0 = 'h', 1 = 'b', 2 = 'c', 3 = '0').
    """
    smile_string = pad_smile(smile_string)
    if preprocess:
        one_hots = [create_one_hot(character_dict.index(char), character_dict) for char in smile_string]
    else:
        one_hots = [create_one_hot(character_dict.index(bytes(char, 'utf-8')), character_dict) for char in smile_string]
    one_hots = np.array(one_hots)
    return one_hots


def create_one_hot(index, character_dict):
    """
    Creates a viable 1D one-hot array where all values are zero but the index
    :param index: index that the letter is
    :param character_dict: list dictionary containing all letters
    :return: 1D one-hot array for the given index
    """
    one_hot = [0 for i in range(len(character_dict))]
    one_hot[index] = 1
    return one_hot


def pad_smile(smile_string):
    """
    Pads a SMILE string so they're all the same length as defined in hyperparameters
    :param smile_string: string to pad
    :return: padded SMILE string (if MAX_LENGTH is 120, it will left adjust the text, then pad with spaces until 120)
    """
    if len(smile_string) == MAX_SMILE_LENGTH:
        return smile_string
    else:
        return smile_string.ljust(MAX_SMILE_LENGTH)


def un_encode(smile_onehots, character_dict):
    """
    Can unencode from one-hots to SMILE strings. Verified as accurate!
    :param smile_onehots: 2D array of one-hot encodings for a given SMILE string
    :param character_dict: list dictionary containing all letters
    :return: the SMILE string represented by the array of one hots
    """
    smile = ""
    for i in range(len(smile_onehots)):
        for j in range(len(smile_onehots[i])):
            if smile_onehots[i][j]:
                smile += character_dict[j]
    return smile


def display_data(dataset):
    '''
    Prints information about the dataset, then the entire dataset for a given name in the h5 file.
    :param dataset: name of dataset to display from h5py
    '''
    h5 = h5py.File(OUT_FILE)
    print(h5[dataset])
    print("")
    print(h5[dataset][:])  # prints whole dataset


if __name__ == "__main__":
    preprocess()
    display_data(DICT_NAME)
    display_data(TRAIN_NAME)
