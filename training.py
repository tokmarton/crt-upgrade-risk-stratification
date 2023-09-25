import os
import pandas as pd
import dill as pickle
from bio_data.bio_data_preprocess import BioDataPreprocess
from model.cross_validated_model import CrossValidatedModel
from utils.utils import get_config

import warnings
warnings.filterwarnings('ignore')


def train_and_validate_internally(data_path: str, config_path: str, target_folder: str):
    """
	Trains a new model and evaluates it internally in a given dataset
	
	Parameters
	----------
	data_path : str
		Path to the CSV file containing the data
		
	config_path : str
		Path to the config file
		
	target_folder : str
		Path to the folder where the results will be saved

    Returns
	-------
    results : pd.DataFrame
        Results of the internal validation
	"""

    # Create target folder
    os.makedirs(target_folder, exist_ok=True)

    # Set path to config file
    config = get_config(config_path)
    param_grid = config['param_grid']

    # Load dataset
    data = pd.read_csv(data_path)

    # Create pipeline for training
    X, y, pipeline = BioDataPreprocess(data,
                                       base_model=config['base_model'],
                                       random_state=config['random_state'],
                                       **config['preprocess']).preprocess_and_create_pipeline()

    # Initialize model
    model = CrossValidatedModel(pipeline, param_grid,
                                scoring=config['scoring'],
                                random_state=config['random_state'],
                                do_feature_selection=config['do_feature_selection'],
                                feature_selection_params=config.get('feature_selection_params', None))

    # Train model and save the trained model
    model.fit_gs(X, y)
    os.makedirs(os.path.join(target_folder, config['model_name']), exist_ok=True)
    pickle.dump(model, open(os.path.join(target_folder, config['model_name'], 'model.pkl'), 'wb'))

    # Calculate performance metrics for internal validation
    cross_val_results = model.cross_validate(X, y)

    # Save results
    results = pd.DataFrame(cross_val_results, index=[config['model_name']])
    results.to_csv(os.path.join(target_folder, config['model_name'], 'cross_val_result.csv'), index=True)
