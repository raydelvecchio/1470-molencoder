import tensorflow as tf
from preprocess import MAX_SMILE_LENGTH


class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.smile_len = MAX_SMILE_LENGTH
        self.chardict_len = 55
        self.latent_size = 292
        self.hidden_dim = 512
        self.encoder_out_size = 435

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, input_shape=(1000, self.smile_len, self.chardict_len), kernel_size=11, activation='swish'),  # just do it
            tf.keras.layers.Conv1D(16, kernel_size=9, activation='swish'),
            tf.keras.layers.Conv1D(8, kernel_size=9, activation='swish'),
            tf.keras.layers.Flatten(),
        ])
        self.decoder = tf.keras.Sequential([
            # Perhaps 292 by 435
            tf.keras.layers.Dense(292, input_shape=(1, 292), activation='swish'),
            tf.keras.layers.GRU(501),
            tf.keras.layers.Dense(self.hidden_dim, activation='swish'),
            tf.keras.layers.Dense(self.chardict_len * self.smile_len)
        ])
        self.mu_layer = tf.keras.layers.Dense(self.latent_size)
        self.logvar_layer = tf.keras.layers.Dense(self.latent_size)

    def call(self, input):
        encoder_out = self.encoder(input)
        dense = tf.keras.layers.Dense(435, input_shape=(540, ), activation='swish')
        encoder_out = dense(encoder_out)

        mu = self.mu_layer(encoder_out)
        logvar = self.logvar_layer(encoder_out)
        latent_rep = self.reparametrize(mu, logvar)

        latent_rep = tf.reshape(latent_rep, (1000, 1, 292))
        decoder_out = self.decoder(latent_rep)
        decoder_out = tf.reshape(decoder_out, (1000, self.smile_len, self.chardict_len))
        softmax = tf.keras.layers.Softmax(axis=2)
        decoder_out = softmax(decoder_out)

        return decoder_out, mu, logvar

    def loss(self, decoder_out, input, mu, logvar):  # implementation of VAE loss; combo of reconstruction and KL
        bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM,)
        reconstruction_loss = bce_loss(input, decoder_out) * input.shape[-1]
        kl_loss = -0.5 * tf.math.reduce_sum(1 + logvar - tf.math.square(mu) - tf.math.exp(logvar))
        return (reconstruction_loss + kl_loss) / input.shape[0]

    def reparametrize(self, mu, logvar):
        epsilon = tf.random.normal((mu.shape[0], mu.shape[1]))
        sigma = tf.math.sqrt(tf.math.exp(logvar))
        return sigma * epsilon + mu
