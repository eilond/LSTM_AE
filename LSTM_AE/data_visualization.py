import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# returns the data as a sequence and each sequences label
def df_to_sequence_input(df, time_steps):
    # print(df.shape)
    subsequences = []
    sub_seq_labels = []
    for i in range(len(df) - time_steps):
        row = df[i: i + time_steps]
        subsequences.append(row)
        label = df[i + time_steps]
        sub_seq_labels.append(label)
    # print(np.array(subsequences).shape)
    return np.array(subsequences), np.array(sub_seq_labels)




if __name__ == "__main__":
    df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')

    amzn_data = df[df['symbol'] == 'AMZN']
    googl_data = df[df['symbol'] == 'GOOGL']
    amzn_daily_high_val = amzn_data['high']
    amzn_data['date'] = pd.to_datetime(amzn_data['date'])
    googl_data['date'] = pd.to_datetime(googl_data['date'])
    googl_daily_high_val = googl_data['high']

    plt.figure(figsize=(14, 7))
    plt.plot(amzn_data['date'], amzn_daily_high_val)
    plt.title('Daily maximal stock price of AMZN stock through 2014-2017')
    plt.xlabel('Date')
    plt.ylabel('Maximum stock price (USD)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(14, 7))
    plt.plot(amzn_data['date'], googl_daily_high_val)
    plt.title('Daily maximal stock price of GOOGL stock through 2014-2017')
    plt.xlabel('Date')
    plt.ylabel('Maximum stock price (USD)')
    plt.grid(True)
    plt.show()
