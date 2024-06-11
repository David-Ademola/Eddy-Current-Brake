import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
N_SAMPLES: int = 500

# Define ranges for each parameter
rotating_disc_thickness = np.random.uniform(0.5, 2.5, N_SAMPLES)  # in cm
disk_radius = np.random.uniform(10, 50, N_SAMPLES)  # in cm
applied_current = np.random.uniform(1, 20, N_SAMPLES)  # in A
number_of_turns = np.random.randint(50, 300, N_SAMPLES)  # number of turns in the coil
air_gap = np.random.uniform(0.1, 1, N_SAMPLES)  # in cm

# Create a DataFrame
data = pd.DataFrame({
    "Rotating Disc Thickness (cm)": rotating_disc_thickness,
    "Disk Radius (cm)": disk_radius,
    "Applied Current (A)": applied_current,
    "Number of Turns": number_of_turns,
    "Air Gap (cm)": air_gap
})

# Constants
K: float = 0.1  # scaling factor

# Generate braking torque using the hypothetical formula
braking_torque = (
    K *
    (1 / data["Air Gap (cm)"]) *
    (data["Applied Current (A)"] ** 2) *
    data["Number of Turns"] *
    data["Disk Radius (cm)"] *
    (1 / data["Rotating Disc Thickness (cm)"])
)

# Add random noise to simulate real-world variations
noise = np.random.normal(0, 10, N_SAMPLES)  # mean 0, std 10
braking_torque += noise

# Add the target variable to the dataset
data["Braking Torque (Nm)"] = braking_torque
# Round all columns in the dataframe to two decimal places
data = data.round(2)

# Display the first few rows of the dataset with the target variable
print(data.head())
data.to_csv("dataset.csv", index=False)  # Save as a CSV file
