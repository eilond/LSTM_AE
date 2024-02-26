import argparse
import pandas as pd
from keras import Input
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from data_visualization import df_to_sequence_input
from keras.layers import LSTM, TimeDistributed, Dense, RepeatVector
from keras.models import Model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--hidden_state_size', type=int, default=64)
args = parser.parse_args()


def make_model(hidden_state_dims, time_steps, data_features):
    input_layer = Input(shape=(time_steps, data_features))
    encoded = LSTM(hidden_state_dims, activation='relu',return_sequences=False)(input_layer)
    decoded = RepeatVector(time_steps)(encoded)
    decoded = LSTM(hidden_state_dims, activation='relu', return_sequences=True)(decoded)
    reconstruction = TimeDistributed(Dense(data_features), name='reconstruction')(decoded)
    prediction = TimeDistributed(Dense(data_features), name='prediction')(decoded[:, -1:, :])  # Assuming prediction structure

    model = Model(inputs=input_layer, outputs=[reconstruction, prediction])
    return model


def create_and_train_lstmae_predictor(time_steps):
    optimizer_class = {
        'Adam': Adam,
    }

    # Load the S&P500 stock price dataset
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
    df_dropped = df.dropna()
    df_filtered = df_dropped[df_dropped['symbol'].isin(['AMZN'])]
    # normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    high_data = scaler.fit_transform(df_filtered[['high']].values)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize variables to track the best model
    best_model = None
    best_loss = float('inf')
    best_history = None
    data_features = high_data.shape[1]

    # Cross-validation loop
    for train_index, test_index in kf.split(high_data):
        # Split data into train and test sets
        X, Y = df_to_sequence_input(high_data[train_index], time_steps)
        X_test, labels = df_to_sequence_input(high_data[test_index], time_steps)

        # Initialize the model
        autoencoder = make_model(hidden_state_dims=args.hidden_state_size, time_steps=time_steps,
                                  data_features=data_features)

        # print(X_train.shape,X_test.shape)

        # Compile the model
        autoencoder.compile(
            optimizer=optimizer_class[args.optimizer](learning_rate=args.learning_rate,
                                                      clipvalue=args.gradient_clipping),
            loss=['mean_squared_error', 'mean_squared_error'],
            )
        # Train the model

        history = autoencoder.fit(X, [X, Y], epochs=args.epochs, batch_size=args.batch_size)
        # for output in autoencoder.outputs:
        #     print(output.name)
        # print(f'metric names:{autoencoder.metrics_names}')

        # Evaluate the model
        loss = autoencoder.evaluate(X_test, [X_test, labels])
        if loss[0] < best_loss:
            best_loss = loss[0]
            best_model = autoencoder
            best_history = history

        return best_model, best_history




if __name__ == "__main__":
    # print(best_history.history.keys())
    best_model, best_history = create_and_train_lstmae_predictor(time_steps=5)

    plt.figure(figsize=(14, 7))
    plt.plot(best_history.history['reconstruction_loss'], label='Reconstruction Loss')
    plt.title(f'training loss vs. time')
    plt.legend()
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(best_history.history['prediction_loss'], label='Prediction Loss')
    plt.title(f'prediction loss vs. time')
    plt.legend()
    plt.show()


