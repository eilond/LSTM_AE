import argparse
import numpy as np
import pandas as pd
# from keras import Input
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from lstm_ae_class import make_predictor_model, Autoencoder
# from data_visualization import df_to_sequence_input
from keras.layers import LSTM, TimeDistributed, Dense, RepeatVector
from keras.models import Model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--gradient_clipping', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--hidden_state_size', type=int, default=64)
args = parser.parse_args()


def df_to_sequence_input(df, time_steps):
    # print(df.shape)
    print(f'df len is{len(df)}')
    subsequences = []
    sub_seq_labels = []
    for i in range(len(df) - time_steps):
        row = df[i: i + time_steps]
        subsequences.append(row)
        label = df[i + time_steps]
        sub_seq_labels.append(label)
    # print(np.array(subsequences).shape)
    return np.array(subsequences), np.array(sub_seq_labels)

def process_data_monthly(df):
    # Ensure 'date' is a datetime column
    df['date'] = pd.to_datetime(df['date'])

    # Create separate year and month columns for grouping
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # Group by symbol, year, and month, then take the first entry of each group
    df_monthly = df.groupby(['symbol', 'year', 'month']).first().reset_index()

    # Make sure 'date' is not duplicated in df_monthly
    if 'date' in df_monthly.columns:
        df_monthly = df_monthly.drop(columns=['date'])

    df_monthly = df_monthly[['symbol', 'high']]
    # df=df[df['symbol'] == 'AMZN']
    df_monthly = df_monthly.dropna()
    return df_monthly


def prepare_data_by_symbol_multi_step(df, N):

    X, Y, symbols = [], [], []
    scaler = MinMaxScaler(feature_range=(0, 1))

    # # Ensure 'date' is a datetime column
    # df['date'] = pd.to_datetime(df['date'])
    #
    # # Create separate year and month columns for grouping
    # df['year'] = df['date'].dt.year
    # df['month'] = df['date'].dt.month
    #
    # # Group by symbol, year, and month, then take the first entry of each group
    # df_monthly = df.groupby(['symbol', 'year', 'month']).first().reset_index()
    #
    # # Make sure 'date' is not duplicated in df_monthly
    # if 'date' in df_monthly.columns:
    #     df_monthly = df_monthly.drop(columns=['date'])
    #
    # df_monthly = df_monthly[['symbol', 'high']]
    # # df=df[df['symbol'] == 'AMZN']
    # df_monthly = df_monthly.dropna()
    df_monthly = process_data_monthly(df)
    # Iterate over each symbol
    for symbol in df_monthly['symbol'].unique():
        symbol_df = df_monthly[df_monthly['symbol'] == symbol]
        scaled_data = scaler.fit_transform(symbol_df[['high']])
        # print(scaled_data)
        if(len(scaled_data) == 48): # take only stock with all the data
            # Create sequences for each symbol
            for i in range(0, len(symbol_df) -1 - N):
                X.append(scaled_data[i :i + N])  # Assuming 'high' is the only feature we're interested in
                Y.append(scaled_data[i + 1 + N])  # Next day's 'high' value as label
                symbols.append(symbol)  # Keep track of the symbol for each sequence

    X, Y = np.array(X), np.array(Y)
    return X, Y, symbols


def create_and_train_lstmae_predictor(time_steps):
    optimizer_class = {
        'Adam': Adam,
    }

    # Load the S&P500 stock price dataset
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
    first_day_each_month, Y, symbols = prepare_data_by_symbol_multi_step(df, time_steps)
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
        X_train, Y_train = first_day_each_month[train_index], Y[train_index]
        X_test, labels = first_day_each_month[test_index], Y[test_index]

        # Initialize the model
        autoencoder = make_predictor_model(hidden_state_dims=args.hidden_state_size, time_steps=time_steps,
                                  data_features=data_features)

        # print(X_train.shape,X_test.shape)

        # Compile the model
        autoencoder.compile(
            optimizer=optimizer_class[args.optimizer](learning_rate=args.learning_rate,
                                                      clipvalue=args.gradient_clipping),
            loss=['mean_squared_error', 'mean_squared_error'],
            )
        # Train the model
        history = autoencoder.fit(X_train, [X_train, Y_train], epochs=args.epochs, batch_size=args.batch_size)
        # Evaluate the model
        loss = autoencoder.evaluate(X_test, [X_test, labels])
        if loss[0] < best_loss:
            best_loss = loss[0]
            best_model = autoencoder
            best_history = history

    return best_model, best_history


if __name__ == "__main__":
    # print(best_history.history.keys())
    best_model, best_history = create_and_train_lstmae_predictor(time_steps=1)

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


