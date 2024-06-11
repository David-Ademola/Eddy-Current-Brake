import pickle

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load the data
dataset = pd.read_csv("dataset.csv")

# Separate features and targets
features = dataset[
    [
        "Rotating Disc Thickness (cm)",
        "Disk Radius (cm)",
        "Applied Current (A)",
        "Number of Turns",
        "Air Gap (cm)",
    ]
]
targets = dataset["Braking Torque (Nm)"]

# Normalize the target variable
target_scaler = MinMaxScaler()
targets = target_scaler.fit_transform(targets.values.reshape(-1, 1))

# Split data into training and testing sets
train_features, test_features, train_targets, test_targets = train_test_split(
    features, targets, test_size=0.1, random_state=42
)

# Scale the features
feature_scaler = StandardScaler()
train_features = feature_scaler.fit_transform(train_features.values)
test_features = feature_scaler.transform(test_features.values)

# Build the model
model = DecisionTreeRegressor()

# Train the model
model.fit(train_features, train_targets)

# Save the model to a file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Evaluate the model
predictions = model.predict(test_features)
mse = mean_squared_error(test_targets, predictions)
mae = mean_absolute_error(test_targets, predictions)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Convert scaled target values to original
predictions = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
test_targets = target_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

# Create a dataframe with predictions and true labels
results = pd.DataFrame(
    {"True Torque (Nm)": test_targets, "Predicted Torque (Nm)": predictions}
)

print(results.head(10))


def objective(params):
    """Function to find optimal torque"""
    disc_thickness, disc_radius, activation_current, num_turns, air_gap = params
    pred_torque = model.predict(
        [[disc_thickness, disc_radius, activation_current, num_turns, air_gap]]
    )
    prediction = np.array(pred_torque[0]).reshape(-1, 1)
    prediction = target_scaler.inverse_transform(prediction)
    return -prediction  # Negative because we want to maximize the torque


# Bounds for the features
bounds = [(0.5, 2.5), (10, 50), (1, 20), (50, 300), (0.1, 1)]

# Transform the bounds using the scaler
scaled_bounds = []

for i, (lower, upper) in enumerate(bounds):
    scaled_lower = feature_scaler.transform(
        np.array([[lower if j == i else 0 for j in range(len(bounds))]])
    )[0][i]
    scaled_upper = feature_scaler.transform(
        np.array([[upper if j == i else 0 for j in range(len(bounds))]])
    )[0][i]
    scaled_bounds.append((scaled_lower, scaled_upper))

# Find the optimal torque using a differential technique
result = differential_evolution(
    objective, bounds, strategy="best1bin", maxiter=1000, popsize=15, tol=0.01
)
optimal_params = result.x  # Get the optimal parameters

# Print the optimal parameters
optimal_params[-1] = round(optimal_params[-1], 2)
columns = features.columns

for i in range(4):
    print(f"{columns[i]}: {round(optimal_params[i])}")
print(f"{columns[-1]}: {optimal_params[-1]}")
