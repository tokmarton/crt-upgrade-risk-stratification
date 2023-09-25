import os
import numpy as np
import pandas as pd
import dill as pickle

import warnings
warnings.filterwarnings('ignore')


def risk_stratify(data_path: str, model_path: str, target_folder: str):
    """
	Evaluates a trained model on a given dataset
	
	Parameters
	----------
	data_path : str
		Path to the CSV file containing the data

	model_path : str
		Path to the trained model

	target_folder : str
		Path to the folder where the results will be saved
	"""

    # Create target folder
    os.makedirs(target_folder, exist_ok=True)

    # Load the trained model
    trained_model = pickle.load(open(model_path, 'rb'))

    # Load data
    data = pd.read_csv(data_path)
    input_features = ['age', 'sex', 'crt_d', 'nyha', 'afib', 'htn', 'diabetes', 'etiology',
                      'mi', 'pci', 'cabg', 'creat', 'gfr', 'lvef', 'lvidd', 'lvids']
    columns_to_drop = np.setdiff1d(data.columns, np.array(input_features))
    X = data.drop(columns_to_drop, axis=1)

    # Run model on the new dataset
    y_proba = trained_model.predict_proba(X)
    y_pred = np.argmax(y_proba, axis=1) + 1

    # Save prediction results
    results = pd.DataFrame({'proba_low_risk': y_proba[:, 0],
                            'proba_intermediate_risk': y_proba[:, 1],
                            'proba_high_risk': y_proba[:, 2],
                            'pred_risk_group': y_pred})
    results = pd.concat([data, results], axis=1)
    results.to_csv(os.path.join(target_folder, model_path.split('\\')[-2],
                                data_path.split('\\')[-1].split('.')[0] + '_with_preds.csv'))
