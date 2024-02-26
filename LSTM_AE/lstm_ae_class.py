from keras.layers import LSTM, TimeDistributed, Dense, RepeatVector
from keras.models import Model


class Autoencoder(Model):
    def __init__(self, hidden_state_dims, time_steps, data_features):
        super(Autoencoder, self).__init__()
        self.time_steps = time_steps
        self.data_features = data_features
        self.hidden_state_dims = hidden_state_dims

        # Encoder reduces the input sequence to a single vector
        self.encoder = LSTM(hidden_state_dims, activation='relu', return_sequences=False)
        # Repeat the encoded vector for the decoder
        self.repeat_vector = RepeatVector(time_steps)
        # Decoder reconstructs the sequence from the encoded vector
        self.decoder = LSTM(hidden_state_dims, activation='relu', return_sequences=True)
        self.time_distributed = TimeDistributed(Dense(data_features))

    def call(self, inputs):
        encoded = self.encoder(inputs)
        repeated = self.repeat_vector(encoded)
        decoded = self.decoder(repeated)
        outputs = self.time_distributed(decoded)
        return outputs
