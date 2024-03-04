import argparse

import numpy as np
import pandas as pd
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lstm_ae_label_prediction import process_data_monthly
from sklearn.model_selection import KFold
from data_visualization import df_to_sequence_input
from lstm_ae_class import Autoencoder


scaler = MinMaxScaler(feature_range=(0, 1))
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--hidden_state_size', type=int, default=64)
args = parser.parse_args()


def prepare_data_by_symbol(df):
    X, symbols = [], []
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_monthly = process_data_monthly(df)
    # Iterate over each symbol
    for symbol in df_monthly['symbol'].unique():
        symbol_df = df_monthly[df_monthly['symbol'] == symbol]
        scaled_data = scaler.fit_transform(symbol_df[['high']])
        # print(scaled_data['high'])
        if(len(scaled_data) == 48): # take only stock with all the data
            X.append(scaled_data)
            symbols.append(symbol)

    return np.array(X), symbols


# to reverse normalization you need to use same Scaler object, so this one uses the global scope variable scaler
def prepare_data_rec(df):
    X, symbols = [], []
    df_monthly = process_data_monthly(df)
    # Iterate over each symbol
    for symbol in df_monthly['symbol'].unique():
        symbol_df = df_monthly[df_monthly['symbol'] == symbol]
        scaled_data = scaler.fit_transform(symbol_df[['high']])
        if(len(scaled_data) == 48): # take only stock with all the data
            X.append(scaled_data)
            symbols.append(symbol)
    return np.array(X), symbols


def create_and_train_lstmae_reconstructor(time_steps):
    optimizer_class = {
        'Adam': Adam,
    }

    # Load the S&P500 stock price dataset
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
    first_day_each_month, symbols = prepare_data_by_symbol(df)
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    # grouped = first_day_each_month.groupby('symbol')
    # # print(grouped)

    # Cross-validation loop
    best_model = None
    best_loss = float('inf')
    best_history = None
    data_features = first_day_each_month[0].shape[1]
    # print(f'high data shape:{high_data.shape}')
    for train_index, test_index in kf.split(first_day_each_month):
        # Split data into train and test sets
        X_train, Y_train = first_day_each_month[train_index], first_day_each_month[train_index]
        X_test, Y_test = first_day_each_month[test_index], first_day_each_month[test_index]

        # Initialize the model
        autoencoder = Autoencoder(hidden_state_dims=args.hidden_state_size, time_steps=time_steps,
                                  data_features=data_features)

        # print(X_train.shape,X_test.shape)

        # Compile the model
        autoencoder.compile(
            optimizer=optimizer_class[args.optimizer](learning_rate=args.learning_rate,
                                                      clipvalue=args.gradient_clipping),
            loss='mse',
            )
        # Train the model
        history = autoencoder.fit(X_train, Y_train, epochs=args.epochs, batch_size=args.batch_size)
        # Evaluate the model
        loss = autoencoder.evaluate(X_test, Y_test)
        if loss < best_loss:
            best_loss = loss
            best_model = autoencoder
            best_history = history

    return best_model, best_history


if __name__ == '__main__':
    reconstructor_model, _ = create_and_train_lstmae_reconstructor(time_steps=48)

    stock_symbols = ['AMZN', 'AAPL', 'MSFT']
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
    df_monthly = process_data_monthly(df)

    for symbol in stock_symbols:
        df_symbol = df[df['symbol'] == symbol]

        stock_sequence, _ = prepare_data_rec(df_symbol)

        reconstructed_data = reconstructor_model.predict(stock_sequence)
        reconstructed_data = reconstructed_data.reshape(-1, 1)
        reconstructed_prices_multi_step = scaler.inverse_transform(reconstructed_data).flatten()


        plt.figure(figsize=(14, 7))
        plt.plot(stock_sequence.flatten(), label='Actual High Prices')
        plt.plot(reconstructed_data, label='Reconstructed High Prices')
        plt.title(f'{symbol} - Actual vs Reconstructed High Prices', fontsize=20)
        plt.xlabel('Time Steps', fontsize=20)
        plt.ylabel('High Prices', fontsize=20)
        plt.legend()
        plt.grid(True)
        plt.show()