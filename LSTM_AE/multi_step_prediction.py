import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler
from data_visualization import df_to_sequence_input
from lstm_ae_label_prediction import create_and_train_lstmae_predictor, process_data_monthly

scaler = MinMaxScaler(feature_range=(0, 1))


def prepare_data_seq(df_monthly, N):
    X, Y, symbols = [], [], []

    for symbol in df_monthly['symbol'].unique():
        symbol_df = df_monthly[df_monthly['symbol'] == symbol]
        scaled_data = scaler.fit_transform(symbol_df[['high']])

        # take only stock with all the data in them
        if len(scaled_data) == 48:
            # Create sequences for each symbol
            for i in range(0, len(symbol_df) - 1 - N):
                X.append(scaled_data[i :i + N])
                Y.append(scaled_data[i + 1 + N])  # Next day's 'high' value as label
                symbols.append(symbol)  # Keep track of the symbol for each sequence

    X, Y = np.array(X), np.array(Y)
    return X, Y, symbols


if __name__ == "__main__":
    # Ensure 'date' is a datetime column
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')

    df_monthly = process_data_monthly(df)
    df_monthly = df_monthly[df_monthly['symbol'] == 'AMZN']

    multi_steps_model, _ = create_and_train_lstmae_predictor(time_steps=24)
    single_step_model, _ = create_and_train_lstmae_predictor(time_steps=1)

    multi_step_sequences, _, _ = prepare_data_seq(df_monthly, N=24)
    single_step_sequences, _, _ = prepare_data_seq(df_monthly, N=1)

    _, reconstructed_multi_step_data = multi_steps_model.predict(multi_step_sequences)
    _, reconstructed_single_step_data = multi_steps_model.predict(single_step_sequences)

    reconstructed_multi_step_data = reconstructed_multi_step_data.reshape(-1, 1)
    reconstructed_prices_multi_step = scaler.inverse_transform(reconstructed_multi_step_data).flatten()

    reconstructed_single_step_data = reconstructed_single_step_data.reshape(-1, 1)
    reconstructed_prices_single_step = scaler.inverse_transform(reconstructed_single_step_data).flatten()

    actual_prices = df_monthly['high'].values
    mid = len(actual_prices) // 2
    actual_prices_ = actual_prices[mid+1:]
    plt.figure(figsize=(14, 7))
    plt.plot(reconstructed_prices_multi_step, label='Reconstructed High Prices')
    plt.plot(actual_prices_, label='Actual High Prices')
    plt.title(f'AMZ stock- Actual vs Predicted High Prices for Single step', fontsize=20)
    plt.xlabel('Time Step', fontsize=20)
    plt.ylabel('High Prices', fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(reconstructed_prices_single_step, label='Reconstructed High Prices')
    plt.plot(actual_prices[:-2], label='Actual High Prices')
    plt.title(f'AMZ stock- Actual vs Predicted High Prices for Single step', fontsize=20)
    plt.xlabel('Time Step', fontsize=20)
    plt.ylabel('High Prices', fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()







