import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler
from data_visualization import df_to_sequence_input
from lstm_ae_label_prediction import create_and_train_lstmae_predictor


df = pd.read_csv('SP 500 Stock Prices 2014-2017.csv')
df_dropped = df.dropna()
df_filtered = df_dropped[df_dropped['symbol'].isin(['AMZN'])]

scaler = MinMaxScaler(feature_range=(0, 1))
high_data = scaler.fit_transform(df_filtered[['high']].values)


multi_steps_model, _ = create_and_train_lstmae_predictor(time_steps=5)
single_step_model, _ = create_and_train_lstmae_predictor(time_steps=1)

multi_step_sequences, _ = df_to_sequence_input(high_data,time_steps=5)
single_step_sequences, _ = df_to_sequence_input(high_data,time_steps=1)

_, reconstructed_multi_step_data = multi_steps_model.predict(multi_step_sequences)
_, reconstructed_single_step_data = multi_steps_model.predict(single_step_sequences)

reconstructed_multi_step_data = reconstructed_multi_step_data.reshape(-1, 1)
reconstructed_prices_multi_step = scaler.inverse_transform(reconstructed_multi_step_data).flatten()

reconstructed_single_step_data = reconstructed_single_step_data.reshape(-1, 1)
reconstructed_prices_single_step = scaler.inverse_transform(reconstructed_single_step_data).flatten()

actual_prices = df_filtered['high'].values.reshape(-1, 1)[5:]

fig, ax = plt.subplots()

ax.plot(reconstructed_prices_multi_step, label='Reconstructed High Prices')
ax.plot(actual_prices, label='Actual High Prices')
ax.set_title(f'AMZ stock- Actual vs Predicted High Prices for Multi step')
ax.set_xlabel('Time Step')
ax.set_ylabel('High Prices')
ax.grid(True)
ax.legend()


ax_inset = fig.add_axes([0.2, 0.47, 0.25, 0.25])  # Position for the inset
ax_inset.set_xlim(250, 300)
ax_inset.set_ylim(350, 400)
ax_inset.plot(reconstructed_prices_multi_step)
ax_inset.plot(actual_prices)
ax_inset.set_title('Magnified', fontsize=8)  # Smaller title font for the inset
# ax_inset.tick_params(labelleft=False, labelbottom=False)  # Remove tick labels
ax_inset.tick_params(labelbottom=False)  # Remove tick labels
ax_inset.grid(True)

# Draw a rectangle on the main plot to indicate the magnified area
rect = Rectangle((250, 350), 50, 50, edgecolor='red', facecolor='none', lw=2)
ax.add_patch(rect)
plt.show()


plt.figure(figsize=(14, 7))
plt.plot(reconstructed_prices_single_step, label='Reconstructed High Prices')
plt.plot(actual_prices, label='Actual High Prices')
plt.title(f'AMZ stock- Actual vs Predicted High Prices for Single step')
plt.xlabel('Time Step')
plt.ylabel('High Prices')
plt.legend()
plt.grid(True)
plt.show()


