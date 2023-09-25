# ML-based risk stratification of patients undergoing CRT upgrade


The primary purpose of this repository is to enable the risk stratification of patients undergoing a cardiac resynchronization therapy (CRT) upgrade procedure using our machine-learning model described in the following paper:

> [**Phenogrouping patients undergoing cardiac resynchronization therapy upgrade using topological data analysis**](https://www.nature.com/srep/)<br/>
  Walter Richard Schwertner, Márton Tokodi, Boglárka Veres, Anett Behon, Eperke Dóra Merkel, Masszi Richárd, Luca Kuthi, Ádám Szijártó, Attila Kovács, István Osztheimer, Endre Zima, László Gellér, Béla Merkely, Annamária Kosztin, Dávid Becker<br/>
  <b>Under Review</b> (2023)

Briefly, we identified 3 phenogroups of patients using topological data analysis based on 16 clinical features. Risk groups with different survival.
Instret figure of TDA and KM curves.
nested cross-validation
using the data of 285 patients who underwent CRT upgrade at the Heart and Vascular Center of Semmelweis University between
Among the evaluated multi-class classifiers, multi-layer perceptron showed the best performance with a balanced accuracy of and ROC of during internal validation.
This repository also contains the scripts used for the training and internal validation.

Tested in an additional 29 patients from an external center -> Patients predicted to be in the high-risk group showed the worst outcomes.

The repository was forked from [szadam96/framework-for-binary-classification](https://github.com/szadam96/framework-for-binary-classification). The upstream repository has been thoroughly described previously:
> [**A machine learning framework for performing binary classification on tabular biomedical data**](https://doi.org/10.1556/1647.2023.00109)<br/>
  Ádám Szijártó, Alexandra Fábián, Bálint Károly Lakatos, Máté Tolvaj, Béla Merkely, Attila Kovács, Márton Tokodi<br/>
  <b>IMAGING</b> (2023)

## Contents of the repository


  - `bio_data` - the collection of functions and classes used for preprocessing
  - `config_files` - a folder containing the configuration files used for training the models
  - `example_data` - a folder containing example data
  - `model` - model classes
  - `trained_models` - a folder containing the trained model(s)
  - `utils` - the collection of functions required for the training and risk stratification scripts
  - `LICENSE.md` - the details of the license (Apache 2.0)
  - `README.md` - a brief explanation of the purpose and content of this repository
  - `main.py` - run this script to train a new model or perform risk stratification
  - `requirements.txt` - the list of the required Python packages
  - `risk_stratification.py` - the scripts required for risk stratifying new patients using the trained model
  - `training.py` - the scripts required for training a new multi-class classifier

## Installation


  1) Clone the repository
  2) Create a virtual environment in Python (version 3.9.13) and activate it
  3) Install the required Python packages (listed in `requirements.txt`) in the virtual environment

## Usage


### Risk stratifying new patients using the trained model

To risk stratify new patients using our model described in the above-referenced [paper](https://www.nature.com/srep/), you should run the following command:

```
python main.py risk_stratify --data PATH_TO_CSV_FILE_WITH_DATA --target_folder PATH_TO_TARGET_FOLDER --model_path PATH_TO_TRAINED_MODEL
```

```PATH_TO_CSV_FILE_WITH_DATA``` is the path to the CSV file containing the data of new patients, ```PATH_TO_TARGET_FOLDER``` is the path to the folder where the prediction results will be saved, and ```PATH_TO_TRAINED_MODEL``` is the path to the trained model. The trained model (`trained_models/mlp/model.pkl`), as well as a CSV file containing the data of three example patients (`example_data/example_data_for_risk_stratification.csv`), are also provided in the repository.

Dictionary of input features:
  - `age`: age at the CRT upgrade procedure (years)
  - `sex`: 0 - male, 1 - female

### Training a new model for multi-class classification

This repository also contains the scripts that we used for training our models. If you want to train a new model for a multi-class classification task, you should run the following command:

```
python main.py train --data PATH_TO_CSV_FILE_WITH_DATA --config_path PATH_TO_CONFIG_FILE --target_folder PATH_TO_TARGET_FOLDER
```

```PATH_TO_CSV_FILE_WITH_DATA``` is the path to the CSV file containing the data of new patients, ```PATH_TO_CONFIG_FILE```
is the path to the file containing the training configurations, ```PATH_TO_TARGET_FOLDER``` is the path to the folder where the prediction results will be saved. Configuration files that we used in our experiments can be found in the `config_files` folder.

For further information, please run the following command: <br>
```
python main.py --help
```

## Contact


For inquiries related to the content of this repository, contact Márton Tokodi, M.D., Ph.D. (tok<!--
-->mar<!--
-->ton[at]gmail.co<!--
-->m).
