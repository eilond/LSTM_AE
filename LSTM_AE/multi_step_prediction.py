import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler
from lstm_ae_label_prediction import create_and_train_lstmae_predictor, process_data_monthly

scaler = MinMaxScaler(feature_range=(0, 1))


def prepare_data_seq(df_monthly, time_steps):
    X, Y, symbols = [], [], []

    for symbol in df_monthly['symbol'].unique():
        symbol_df = df_monthly[df_monthly['symbol'] == symbol]
        scaled_data = scaler.fit_transform(symbol_df[['high']])

        # take only stock with all the data in them
        if len(scaled_data) == 48:
            # Create sequences for each symbol
            for i in range(0, len(symbol_df) - 1 - time_steps):
                X.append(scaled_data[i :i + time_steps])
                Y.append(scaled_data[i + 1 + time_steps])  # Next day's 'high' value as label
                symbols.append(symbol)  # Keep track of the symbol for each sequence

    X, Y = np.array(X), np.array(Y)
    return X, Y, symbols


if __name__ == "__main__":
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')

    df_monthly = process_data_monthly(df)
    df_monthly = df_monthly[df_monthly['symbol'] == 'AMZN']

    multi_steps_model, _ = create_and_train_lstmae_predictor(time_steps=24)
    single_step_model, _ = create_and_train_lstmae_predictor(time_steps=1)

    multi_step_sequences, _, _ = prepare_data_seq(df_monthly, time_steps=24)
    single_step_sequences, _, _ = prepare_data_seq(df_monthly, time_steps=1)

    _, reconstructed_multi_step_data = multi_steps_model.predict(multi_step_sequences)
    _, reconstructed_single_step_data = multi_steps_model.predict(single_step_sequences)

    reconstructed_multi_step_data = reconstructed_multi_step_data.reshape(-1, 1)
    reconstructed_prices_multi_step = scaler.inverse_transform(reconstructed_multi_step_data).flatten()

    reconstructed_single_step_data = reconstructed_single_step_data.reshape(-1, 1)
    reconstructed_prices_single_step = scaler.inverse_transform(reconstructed_single_step_data).flatten()

    actual_prices = df_monthly['high'].values
    mid = len(actual_prices) // 2 - 1
    rec_list = list(reconstructed_prices_multi_step)
    rec_list = [None] * mid + rec_list

    fig, ax = plt.subplots()

    ax.plot(rec_list, label='Reconstructed High Prices')
    ax.plot(actual_prices[:-2], label='Actual High Prices')
    ax.set_title(f'AMZN stock- Actual vs Predicted High Prices for Multi step')
    ax.set_xlabel('Sample Number (per month)', fontsize=20)
    ax.set_ylabel('High Prices', fontsize=20)
    ax.grid(True)
    ax.legend()
    ax_inset = fig.add_axes([0.2, 0.47, 0.25, 0.25])  # Position for the inset
    ax_inset.set_xlim(38, 42)
    ax_inset.set_ylim(930, 980)
    ax_inset.plot(rec_list)
    ax_inset.plot(actual_prices[:-2])
    ax_inset.set_title('Magnified', fontsize=8)  # Smaller title font for the inset
    ax_inset.tick_params(labelbottom=False)  # Remove tick labels
    ax_inset.grid(True)
    # Draw a rectangle on the main plot to indicate the magnified area
    rect = Rectangle((38, 930), 4, 50, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(rect)
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(reconstructed_prices_single_step, label='Reconstructed High Prices')
    plt.plot(actual_prices[:-2], label='Actual High Prices')
    plt.title(f'AMZN stock- Actual vs Predicted High Prices for Single step', fontsize=20)
    plt.xlabel('Sample Number (per month)', fontsize=20)
    plt.ylabel('High Prices', fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.show()







