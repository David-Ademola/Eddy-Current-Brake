import numpy as np
import pandas as pd

# Given parameters
min_current, max_current, current_step = 0.5, 5, 0.25
min_omega, max_omega, omega_step = 0, 3000, 10

# Constants
mu_0 = 4 * np.pi * 10**-7  # Permeability of free space (H/m)
N_TURNS = 205  # Number of turns per pole
GAP_WIDTH = 2 * 10**-3  # Air-gap width (m)
SIGMA = 18560000  # Conductivity of disk (S/m)
RADIUS = 0.02  # RADIUS of electromagnet (m)
THICKNESS = 0.01  # THICKNESS of disk (m)
k = 1  # Constant, adjust based on empirical data


# Function to calculate torque
def calculate_torque(input_current: float, shaft_speed: float) -> float:
    mag_flux_density = (
        mu_0 * N_TURNS * input_current
    ) / GAP_WIDTH  # Magnetic flux density
    shaft_speed_rads = shaft_speed * 2 * np.pi / 60  # Convert RPM to rad/s
    torque = (
        k * (mag_flux_density*2 * SIGMA * RADIUS*2 * shaft_speed_rads) / THICKNESS
    )  # Torque

    return torque


# Generate the dataset
data = [
    [current, omega_rpm, calculate_torque(current, omega_rpm)]
    for current in np.arange(min_current, max_current + current_step, current_step)
    for omega_rpm in np.arange(min_omega, max_omega + omega_step, omega_step)
]

# Convert to dataframe
dataframe = pd.DataFrame(
    data, columns=["Excitation Current (A)", "Shaft Speed (RPM)", "Torque (Nm)"]
)

# Display the first 10 rows
print(dataframe.head(50))

# Save DataFrame to CSV file
csv_file_path = "eddy_current_brake_dataset.csv"
dataframe.to_csv(csv_file_path, index=False)

print(f"DataFrame saved as {csv_file_path}")
