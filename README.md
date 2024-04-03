# ITC 6103 – APPLIED MACHINE LEARNING Project

## Project Structure

This project requires specific data files and Python scripts organized in a particular directory structure for running machine learning tasks:

### Data Files

Ensure to download and organize your data files as follows:

- **US Census Data (1990):** Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29). Place the extracted file in the `usCensusData` directory and rename it to `USCensus1990.data.txt`.

- **Regression Dataset:** The `car_prices.csv` file should be placed in the `regressionDataset` folder.

- **Training Data:** Extract `train.zip` and place its contents into the `train` folder.

### Python Scripts

Three main Python scripts are provided for analysis:

- `q1Clustering.py`: For clustering analysis.
- `q2Regression.py`: For regression analysis.
- `q3Prediction.py`: For making predictions.

## Setup Instructions

### Installing Python Libraries

Please install the required libraries.

## Data Preparation

### US Census Data (1990):

- Go to the [UCI Machine Learning Repository's page for US Census Data (1990)](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29).
- Download the dataset and extract it.
- Place `USCensus1990.data.txt` in the `./usCensusData` directory.

### Regression Dataset:

- The `car_prices.csv` file should already be located in the `./regressionDataset`.

### Training Data:

- Unzip the provided `train.zip` file.
- Move its contents to the `./train` directory.

## Running the Scripts

Execute the scripts from the project's root directory to perform the analyses:

- For clustering analysis:
  ```bash
  python q1Clustering.py
- For regression  analysis:
  ```bash
  python q2Regression.py
- For making prediction:
  ```bash
  python q3Prediction.py