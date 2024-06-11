# Braking Torque Prediction and Optimization

This repository contains a project aimed at predicting and optimizing the braking torque of a system using machine learning and optimization techniques. The project includes two main components: data generation and model training and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview

The goal of this project is to predict the braking torque based on various input features and find the optimal set of parameters that maximize the braking torque using a machine learning model and an optimization algorithm. The project uses a decision tree regressor for prediction and a differential evolution algorithm for optimization.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/David-Ademola/Eddy-Current-Brake
    cd Eddy-Current-Brake
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Generate the dataset**:
    Run `dataset.py` to generate a synthetic dataset and save it as `dataset.csv`.
    ```sh
    python dataset.py
    ```

2. **Train the model and optimize the parameters**:
    Run `main.py` to train the decision tree model, evaluate its performance, and find the optimal parameters.
    ```sh
    python main.py
    ```

    The script will output the mean squared error (MSE) and mean absolute error (MAE) of the model, display the first few predictions, and print the optimal parameters for maximizing the braking torque.

## Files

- `main.py`: The main script for loading the dataset, training the model, evaluating its performance, and optimizing the parameters.
- `dataset.py`: The script for generating a synthetic dataset with input features and target braking torque values.
- `dataset.csv`: The generated dataset file (created by running `dataset.py`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to modify and expand this project to suit your needs. Contributions are welcome!

---

**Note**: The parameters and ranges used in this project are hypothetical and should be adjusted based on the actual application and data characteristics.