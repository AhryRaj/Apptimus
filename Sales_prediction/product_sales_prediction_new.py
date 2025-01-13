import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle
import keras_tuner as kt
from Data_preprocessing import pivot_data

# Scale data for LSTM (normalizing values between 0 and 1)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(pivot_data)
print(scaled_data)

# Create sequences
sequence_length = 12
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Check if a saved model exists
model_path = './models/best_sales_forecasting_model_new.pkl'
if os.path.exists(model_path):
    # Load the model and scaler
    with open(model_path, 'rb') as file:
        saved_data = pickle.load(file)
        best_model = saved_data['model']
        scaler = saved_data['scaler']
    print("Loaded the saved model successfully.")
else:
    print("No saved model found. Training a new model.")

    # Function to build the model for Keras Tuner
    def build_model(hp):
        model = Sequential()

        # First LSTM layer with tunable units and dropout
        model.add(LSTM(units=hp.Int('units_lstm_1', min_value=32, max_value=128, step=16),
                       return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)))

        # Second LSTM layer with tunable units and dropout
        model.add(LSTM(units=hp.Int('units_lstm_2', min_value=32, max_value=128, step=16), return_sequences=True))
        model.add(Dropout(hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)))

        # Third LSTM layer (optional, can be removed depending on performance)
        model.add(LSTM(units=hp.Int('units_lstm_3', min_value=32, max_value=128, step=16)))
        model.add(Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.5, step=0.1)))

        # Dense layer to output the number of product types
        model.add(Dense(y_train.shape[1]))

        # Tunable learning rate for the optimizer
        model.compile(optimizer=Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='LOG', default=1e-3)),
            loss='mse')

        return model

    # Initialize the Keras Tuner with Bayesian Optimization
    tuner_new = kt.BayesianOptimization(
        build_model,
        objective='val_loss',
        max_trials=80,
        directory='./models/tuner_new',
        project_name='sales_forecasting_bayes')

    # Add EarlyStopping to stop training if no improvement for 5 epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Perform hyperparameter search
    tuner_new.search(X_train, y_train, epochs=30, batch_size=64,
                 validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Get the best model
    best_model = tuner_new.get_best_models(num_models=1)[0]

    # Save the best model and scaler
    model_path = './models/best_sales_forecasting_model_new.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump({'model': best_model, 'scaler': scaler}, file)

# Evaluate the best model
test_loss = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")

# Make predictions on the test set
predictions = best_model.predict(X_test)

# Reverse scaling to get actual values
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

# Calculate R² score
r2 = r2_score(y_test_actual.flatten(), predictions_actual.flatten())
print(f"Model Accuracy (R² Score): {r2:.2f}")

# Plot actual vs predicted values for the test set
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual[:, 0], label='Actual Sales')
plt.plot(predictions_actual[:, 0], label='Predicted Sales')
plt.title('Test Set: Actual vs Predicted Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Predict the next 12 months
future_input = scaled_data[-sequence_length:]
future_predictions = []

for _ in range(12):
    pred = best_model.predict(future_input[np.newaxis, :, :])
    future_predictions.append(pred[0])
    future_input = np.vstack([future_input[1:], pred])

# Convert future predictions back to actual values
future_predictions_actual = scaler.inverse_transform(future_predictions)

# Create a DataFrame for future predictions
future_dates = pd.date_range(start=pivot_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='ME')
future_df = pd.DataFrame(future_predictions_actual, index=future_dates, columns=pivot_data.columns)

# Save the future predictions to a .csv file
future_df.to_csv('./data/future_sales_predictions_new.csv')

# Display future predictions
print("Future Sales Predictions:")
print(future_df)

# Plot future sales predictions
future_df.plot(figsize=(12, 6), title='Future Sales Predictions', xlabel='Time', ylabel='Sales')
plt.show()
