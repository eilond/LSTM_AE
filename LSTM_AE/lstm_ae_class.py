from keras import Input
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


def make_predictor_model(hidden_state_dims, time_steps, data_features):
    input_layer = Input(shape=(time_steps, data_features))
    encoded = LSTM(hidden_state_dims, activation='relu', return_sequences=False)(input_layer)
    decoded = RepeatVector(time_steps)(encoded)
    decoded = LSTM(hidden_state_dims, activation='relu', return_sequences=True)(decoded)
    reconstruction = TimeDistributed(Dense(data_features), name='reconstruction')(decoded)
    prediction = TimeDistributed(Dense(data_features), name='prediction')(decoded[:, -1:, :])

    model = Model(inputs=input_layer, outputs=[reconstruction, prediction])
    return model