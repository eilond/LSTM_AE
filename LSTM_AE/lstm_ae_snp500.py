import argparse
import pandas as pd
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from data_visualization import df_to_sequence_input
from lstm_ae_class import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--hidden_state_size', type=int, default=64)
args = parser.parse_args()

optimizer_class = {
    'Adam': Adam,
}

# Load the S&P500 stock price dataset
df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
df_dropped = df.dropna()


stock_symbols = ['AMZN', 'AAPL', 'MSFT']
for symbol in stock_symbols:
    df_filtered = df_dropped[df_dropped['symbol'].isin([symbol])]
    time_steps = 5

    scaler = MinMaxScaler(feature_range=(0, 1))
    high_data = scaler.fit_transform(df_filtered[['high']].values)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize variables to track the best model
    best_model = None
    best_loss = float('inf')

    # Define time_steps and data_features
    data_features = high_data.shape[1]

    # Cross-validation loop
    for train_index, test_index in kf.split(high_data):
        # Split data into train and test sets
        # labels are for next phase, can be ignored ATM
        X_train, labels = df_to_sequence_input(high_data[train_index], time_steps)
        X_test, labels = df_to_sequence_input(high_data[test_index], time_steps)

        # Initialize the model
        autoencoder = Autoencoder(hidden_state_dims=args.hidden_state_size, time_steps=time_steps,
                                  data_features=data_features)

        # print(X_train.shape,X_test.shape)

        # Compile the model
        autoencoder.compile(
            optimizer=optimizer_class[args.optimizer](learning_rate=args.learning_rate,
                                                      clipvalue=args.gradient_clipping),
            loss='mean_squared_error')
        # Train the model
        autoencoder.fit(X_train, X_train, epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=(X_test, X_test))
        # Evaluate the model
        loss = autoencoder.evaluate(X_test, X_test)
        if loss < best_loss:
            best_loss = loss
            best_model = autoencoder

    # plotting the model's predictions for the stock
    high_data = df_filtered['high'].values.reshape(-1, 1)
    high_data_scaled = scaler.transform(high_data)
    high_data_subsequences, labels = df_to_sequence_input(high_data_scaled, time_steps)

    # Predict the reconstructed sequences using the best model
    reconstructed = best_model.predict(high_data_subsequences)
    reconstructed = reconstructed[0::time_steps]
    reconstructed = reconstructed.reshape(-1, 1)
    reconstructed_prices = scaler.inverse_transform(reconstructed).flatten()
    # print(f"the length of reconstructed is: {len(reconstructed_prices)}\n the length of prices is: {len(high_data[time_steps-1:])}")
    # Actual prices for the corresponding timestamps
    actual_prices = high_data

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(actual_prices, label='Actual High Prices')
    plt.plot(reconstructed_prices, label='Reconstructed High Prices')
    plt.title(f'{symbol} - Actual vs Reconstructed High Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('High Prices')
    plt.legend()
    plt.grid(True)
    plt.show()




