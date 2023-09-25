from pathlib import Path
import numpy as np
import scipy
import yaml
from model.other_models import CalibratedRF, CalibratedXGBoost, CalibratedLog, CalibratedKNN, CalibratedSVC, \
    CalibratedMLP, CalibratedGBC


def mean_confidence_interval(data, confidence=0.95):
    """
	Calculates the mean and the confidence interval
	
	Parameters
	----------
	data : array-like
		The data that is used for the calculation
	
	confidence : float
		The confidence value that is used for the confidence interval calculation
		
	Returns
	-------
	mean : float
		The mean of the data
	
	lower : float
		The lower limit of the confidence interval
		
	upper : float
		The upper limit of the confidence interval
	"""

    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

    lower = m - h
    upper = m + h

    return m, lower, upper


def get_param_grid_from_config(param_grid: dict, model_name: str):
    """
	Creates a hyperparameter grid that can be passed to the grid search algorithm for hyperparameter optimization
	
	Parameters
	----------
	param_grid : dict
		Dictionary containing the hyperparameters

	model_name : str
	    Name of the model

	Returns
	-------
	Dictionary containing the hyperparameter grid
	"""

    if param_grid is not None:
        param_grid = {f'classifier__{k}': v for k, v in param_grid.items()}
    else:
        with open('./utils/default_hyperparams.yaml', 'r') as f:
            default_params = yaml.safe_load(f)
            param_grid = {f'classifier__{k}': v for k, v in default_params[model_name].items()}

    return param_grid


def get_model(model_name, cv, n_jobs):
    """
	Initializes a multiclass classifier

	Parameters
	----------
	model_name : str
		Name of the classifier
	
	cv : StratifiedKFold
		Cross-validation instance

	n_jobs : int
		Number of jobs running parallelly

	Returns
	-------
	A calibrated multiclass classifier
	"""

    if model_name == 'randomforest':
        return CalibratedRF(cv=cv, n_jobs=n_jobs)
    if model_name == 'xgboost':
        return CalibratedXGBoost(cv=cv, n_jobs=n_jobs)
    if model_name == 'log_l1':
        return CalibratedLog(penalty='l1', cv=cv, n_jobs=n_jobs)
    if model_name == 'log_l2':
        return CalibratedLog(penalty='l2', cv=cv, n_jobs=n_jobs)
    if model_name == 'knn':
        return CalibratedKNN(cv=cv, n_jobs=n_jobs)
    if model_name == 'svc':
        return CalibratedSVC(cv=cv, n_jobs=n_jobs)
    if model_name == 'mlp':
        return CalibratedMLP(cv=cv, n_jobs=n_jobs)
    if model_name == 'gbc':
        return CalibratedGBC(cv=cv, n_jobs=n_jobs)

    raise ValueError(f'model {model_name} is not supported!')


def get_config(config_path: str):
    """
	Import training configurations

	Parameters
	----------
	config_path : str
		Path to the YAML file containing the configurations

	Returns
	-------
	configs : dict
		The imported configurations
	"""

    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    configs = dict()

    configs['cv'] = config.get('cv', 5)
    configs['random_state'] = config.get('random_state', 42)
    configs['n_jobs'] = config.get('n_jobs', -1)
    configs['do_feature_selection'] = config.get('do_feature_selection', False)

    param_grid = config.get('hyperparameters', None)
    model_name = config.get('model', None)
    scoring = config.get('scoring', 'balanced_accuracy')
    model = get_model(model_name, configs['cv'], configs['n_jobs'])
    configs['model_name'] = model_name
    configs['scoring'] = scoring
    configs['base_model'] = model
    configs['param_grid'] = get_param_grid_from_config(param_grid, model_name)

    preproc = config['preprocess']
    configs['balancing_method'] = preproc.get('balancing_method', None)
    configs['weighted'] = False
    if configs['balancing_method'] == 'weighted':
        configs['weighted'] = True

    configs['preprocess'] = dict()
    configs['preprocess']['smote'] = False
    if configs['balancing_method'] == 'smote':
        configs['preprocess']['smote'] = True
    configs['preprocess']['target_column'] = preproc['target_column']
    configs['preprocess']['normalizer'] = preproc.get('normalizer', None)
    configs['preprocess']['drop_threshold'] = preproc.get('drop_threshold', 0.3)
    configs['preprocess']['categorical_impute'] = preproc.get('categorical_impute', 'most_frequent')
    configs['preprocess']['continuous_impute'] = preproc.get('continuous_impute', 'mean')

    if configs['do_feature_selection']:
        configs['feature_selection_params'] = config['feature_selection_params']

    return configs
