# ML-based risk stratification of patients undergoing CRT upgrade


The primary purpose of this repository is to enable the risk stratification of patients without undergoing a cardiac resynchronization therapy (CRT) upgrade procedure using our machine-learning model described in the following paper:

> [**Phenogrouping patients undergoing cardiac resynchronization therapy upgrade using topological data analysis**](https://www.nature.com/srep/)<br/>
  Walter Richard Schwertner, Márton Tokodi, Boglárka Veres, Anett Behon, Eperke Dóra Merkel, Masszi Richárd, Luca Kuthi, Ádám Szijártó, Attila Kovács, István Osztheimer, Endre Zima, László Gellér, Béla Merkely, Annamária Kosztin, Dávid Becker<br/>
  <b>Under Review</b> (2023)

Briefly, we identified 3 phenogroups of patients using topological data analysis based on 16 clinical features . Risk groups with different survival.
After the topological network was created, we wanted to divide it into regions with different clinical characteristics and risks of all-cause mortality. The defined regions distinguish patient phenogroups with a specific risk of mortality. By this, we can compare the all-cause mortality of the risk groups following CRT-D or CRT-P upgrade.
To this end, we first performed community autogrouping using the Louvain method to find the best possible grouping of nodes with high intra- but low inter-group connectivity.(24) With this algorithm, we generated 14 autogroups, which were then sorted based on the survival rate of their members to identify the group with the lowest and the highest mortality rate. Next, each group was merged with an adjacent group exhibiting the most similar mortality rate. This step was repeated multiple times until three risk groups (low-risk, intermediate-risk, and high-risk groups) with a nearly equal number of patients were created (Supplementary Figure 1).
326 patients prior ventricular arrhythmias or previously implanted implantable cardioverter-defibrillator 
The application of TDA and autogrouping resulted in the formation of a looped network in which the low-risk and high-risk regions were located at opposite poles (Figure 3). These two regions were conjoined by sections containing patients with an intermediate risk of death on both the lower and upper arc of the loop. The combination of the two intermediate-risk areas is referred to as the intermediate-risk group throughout the manuscript.
were also significant differences in the survival of the risk groups 
Instret figure of TDA and KM curves.
nested cross-validation (with 5 folds in both the inner and outer loops)
using the data of 285 patients who underwent CRT upgrade at the Heart and Vascular Center of Semmelweis University between 2001 and 2020
Among the evaluated multi-class classifiers, multi-layer perceptron showed the best performance with a balanced accuracy of and ROC of during internal validation.
This repository also contains the scripts used for the training and internal validation.

The best-performing model was also tested on an additional 29 patients from an external center. In this cohort, all patients who were predicted to belong to the high-risk phenogroup died within 10 years following the upgrade procedure (<b>Figure 2</b>).

The repository was forked from [szadam96/framework-for-binary-classification](https://github.com/szadam96/framework-for-binary-classification), which we have thoroughly described in this paper:
> [**A machine learning framework for performing binary classification on tabular biomedical data**](https://doi.org/10.1556/1647.2023.00109)<br/>
  Ádám Szijártó, Alexandra Fábián, Bálint Károly Lakatos, Máté Tolvaj, Béla Merkely, Attila Kovács, Márton Tokodi<br/>
  <b>IMAGING</b> (2023)

## Contents of the repository


  - `bio_data` - the collection of functions and classes used for preprocessing
  - `config_files` - the folder containing the configuration files used for training the models
  - `example_data` - the folder containing example data
  - `figs` - the folder containing the figures used in `README.md`
  - `model` - model classes
  - `trained_models` - the folder containing the trained model(s)
  - `utils` - the collection of functions required for the training and risk stratification scripts
  - `LICENSE.md` - the details of the license (Apache 2.0)
  - `README.md` - the brief explanation of the purpose and content of this repository
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

To risk stratify new patients using our model (i.e., an ensemble of 5 multi-layer perceptrons) described in the above-referenced [paper](https://www.nature.com/srep/), you should run the following command:

```
python main.py risk_stratify --data PATH_TO_CSV_FILE_WITH_DATA --target_folder PATH_TO_TARGET_FOLDER --model_path PATH_TO_TRAINED_MODEL
```

```PATH_TO_CSV_FILE_WITH_DATA``` is the path to the CSV file containing the data of new patients, ```PATH_TO_TARGET_FOLDER``` is the path to the folder where the prediction results will be saved, and ```PATH_TO_TRAINED_MODEL``` is the path to the trained model. The trained model (`trained_models/mlp/model.pkl`), as well as a CSV file containing the data of three example patients (`example_data/example_data_for_risk_stratification.csv`), are also provided in the repository. You should use the provided template (`example_data/template.csv`) without changing the order and names of the columns. Additional information on the columns (i.e., input features) can be found below.

Dictionary of input features:
  1) `age`: age at the CRT upgrade procedure (years)
  2) `sex`: sex of the patient (0 - male, 1 - female)
  3) `crt_d`: type of the implanted device (0 - CRT-P, 1 - CRT-D)
  4) `nyha`: NYHA functional class prior to the upgrade procedure (1 - I, 2 - II, 3 - III, 4 - IV)
  5) `afib`: history of atrial fibrillation (0 - no, 1 - yes)
  6) `htn`: hypertension (0 - no, 1 - yes)
  7) `diabetes`: diabetes mellitus (0 - no, 1 - yes)
  8) `etiology`: etiology of heart failure (0 - non-ischemic, 1 - ischemic)
  9) `mi`: history of myocardial infarction (0 - no, 1 - yes)
  10) `pci`: history of percutaneous coronary intervention (0 - no, 1 - yes)
  11) `cabg`: history of coronary artery bypass graft surgery (0 - no, 1 - yes)
  12) `creat`: serum creatinine (µmol/L)
  13) `gfr`: glomerular filtration rate calculated based on the MDRD formula (mL/min/1.73 m<sup>2</sup>)
  14) `lvef`: left ventricular ejection fraction measured using echocardiography (%)
  15) `lvidd`: left ventricular internal diameter at end-diastole measured using echocardiography (mm)
  16) `lvids`: left ventricular internal diameter at end-systole measured using echocardiography (mm)

### Training a new model for multi-class classification

This repository also contains the scripts that we used for training our models. If you want to train a new model for a multi-class classification task, you should run the following command:

```
python main.py train --data PATH_TO_CSV_FILE_WITH_DATA --config_path PATH_TO_CONFIG_FILE --target_folder PATH_TO_TARGET_FOLDER
```

```PATH_TO_CSV_FILE_WITH_DATA``` is the path to the CSV file containing the data of new patients, ```PATH_TO_CONFIG_FILE```
is the path to the file containing the training configurations, ```PATH_TO_TARGET_FOLDER``` is the path to the folder where the prediction results will be saved. Configuration files that we used in our experiments can be found in the `config_files` folder.

For further information on the usage of our codes, please run the following command: <br>
```
python main.py --help
```

## Contact


For inquiries related to the content of this repository, contact Márton Tokodi, M.D., Ph.D. (tok<!--
-->mar<!--
-->ton[at]gmail.co<!--
-->m).
