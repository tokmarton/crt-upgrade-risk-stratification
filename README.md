# ML-based risk stratification of patients undergoing CRT upgrade


The primary purpose of this repository is to enable the risk stratification of patients undergoing a cardiac resynchronization therapy upgrade procedure using our machine-learning model described in the following paper:

> [**Phenogrouping patients undergoing cardiac resynchronization therapy upgrade using topological data analysis**](https://www.nature.com/srep/)<br/>
  Walter Richard Schwertner, Márton Tokodi, Boglárka Veres, Anett Behon, Eperke Dóra Merkel, Masszi Richárd, Luca Kuthi, Ádám Szijártó, Attila Kovács, István Osztheimer, Endre Zima, László Gellér, Béla Merkely, Annamária Kosztin, Dávid Becker<br/>
  <b>Under Review</b>

This repository enables contains the codes to train and validate the training and validation of machine-learning models for multi-class classification. 

The repository was forked from `szadam96\framework-for-binary-classification`. The upstream repository has been thoroughly described previously:
> [**A machine learning framework for performing binary classification on tabular biomedical data**](https://doi.org/10.1556/1647.2023.00109)<br/>
  Ádám Szijártó, Alexandra Fábián, Bálint Károly Lakatos, Máté Tolvaj, Béla Merkely, Attila Kovács, Márton Tokodi<br/>
  <b>IMAGING</b> (2023)

## Installation


  1) Clone the repository
  2) Create a virtual environment in Python (version 3.9.13) and activate it
  3) Install the required Python packages (listed in `requirements.txt`) in the virtual environment
     ```
     pip install -r requirements.txt
     ```

## Usage


### Risk stratifying new patients using the trained model

To risk stratify new patients using our model described in the above-referenced [paper](https://www.nature.com/srep/), you should run the following command:
```
python main.py risk_stratify --data PATH_TO_CSV_FILE_WITH_DATA --target_folder PATH_TO_TARGET_FOLDER --model_path PATH_TO_TRAINED_MODEL
```

```PATH_TO_CSV_FILE_WITH_DATA``` is the path to the CSV file containing the data of new patients, ```PATH_TO_TARGET_FOLDER``` is the path to the folder where the prediction results will be saved, and ```PATH_TO_TRAINED_MODEL``` is the path to the trained model. The trained model can be found in `trained_models\mlp\model.pkl` and a CSV file containing the data of three example patients in `example_data\example_data_for_risk_stratification.csv`.

### Training a new model for multi-class classification

This repository also contains the codes that we used for training our models. If you want to train a new model for a multi-class classification task, you should run the following command:
```
python main.py train --data PATH_TO_CSV_FILE_WITH_DATA --config_path PATH_TO_CONFIG_FILE --target_folder PATH_TO_TARGET_FOLDER
```

```PATH_TO_CSV_FILE_WITH_DATA``` is the path to the CSV file containing the data of new patients, ```PATH_TO_CONFIG_FILE```
is the path to the file containing the training configurations, ```PATH_TO_TARGET_FOLDER``` is the path to the folder where the prediction results will be saved. Configuration files that we used in our experiments can be found in the `config_files' folder.

For further information, please run the following command: <br>
```
python main.py --help
```

## Contact


For inquiries related to the content of this repository, contact Márton Tokodi, M.D., Ph.D. (tok<!--
-->mar<!--
-->ton[at]gmail.co<!--
-->m).
