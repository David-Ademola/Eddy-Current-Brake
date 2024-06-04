import pandas as pd
from keras import Sequential
from keras.backend import clear_session
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the data
dataframe = pd.read_csv("eddy_current_brake_dataset.csv")

# Separate features and targets
features = dataframe[["Excitation Current (A)", "Shaft Speed (RPM)"]]
targets = dataframe["Torque (Nm)"]

# Normalize the target variable
target_scaler = MinMaxScaler()
targets = target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()

# Split data into training and testing sets
train_features, test_features, train_targets, test_targets = train_test_split(
    features, targets, test_size=0.1, random_state=42
)

# Scale the features
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features)
test_features = feature_scaler.transform(test_features)

# Build the neural network
clear_session()
model = Sequential(
    [
        Dense(128, "relu", input_shape=(2,)),
        Dense(128, "relu"),
        Dense(1, "linear")
    ]
)
# Compile the model
model.compile("adam", "mse", ["mae"])


# Learning rate scheduler
def scheduler(epoch: int, lr: float) -> float:
    return lr if epoch < 50 else lr * 0.99


scheduler = LearningRateScheduler(scheduler)
stopper = EarlyStopping("val_mae", patience=10, restore_best_weights=True)
# Train the model
history = model.fit(train_features, train_targets, epochs=200, callbacks=[scheduler, stopper], validation_split=0.2)
print()

# Plot the training history
pyplot.figure(figsize=(14, 6))

# Plot loss and val_loss
pyplot.subplot(1, 2, 1)
pyplot.plot(history.history["loss"], label="loss")
pyplot.plot(history.history["val_loss"], label="val_loss")
pyplot.xlabel("Epochs")
pyplot.ylabel("Loss")
pyplot.title("Model Loss")
pyplot.legend()

# Plot mae and val_mae
pyplot.subplot(1, 2, 2)
pyplot.plot(history.history["mae"], label="mae")
pyplot.plot(history.history["val_mae"], label="val_mae")
pyplot.xlabel("Epochs")
pyplot.ylabel("Mean Absolute Error")
pyplot.title("Model Mean Absolute Error")
pyplot.legend()

pyplot.tight_layout()
pyplot.savefig("training_history.png")  # Save the plot
pyplot.show()

# Save the model
model.save("EddyCurrentBrakeModel.keras")
print("Model saved successfully \n")

# Evaluate the model
loss, mae = model.evaluate(test_features, test_targets)
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Loss: {loss:.4f} \n")

# Make predictions
predictions = model.predict(test_features)

# Convert scaled target values to original
predictions = target_scaler.inverse_transform(predictions).flatten()
test_targets = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

# Create a dataframe with predictions and true labels
results = pd.DataFrame({"True Torque (Nm)": test_targets, "Predicted Torque (Nm)": predictions})
# Save the dataframe as a csv file
results.to_csv("results.csv")

# Print the first 20 rows
print(results.head(20))
