import numpy as np
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


def get_normalizer(normalizer_name: str):
	"""
	Initializes a normalizer instance

	Parameters
	----------
	normalizer_name : str
		Name of the normalizer

	Returns
	-------
	normalizer : object
		Normalizer object
	"""

	if normalizer_name == 'standard':
		return StandardScaler()
	elif normalizer_name == 'l1':
		return Normalizer(norm='l1')
	elif normalizer_name == 'l2':
		return Normalizer(norm='l2')
	elif normalizer_name == 'minmax':
		return MinMaxScaler()
	elif normalizer_name == 'robust':
		return RobustScaler()
	elif normalizer_name == '':
		return FunctionTransformer()
	else:
		raise ValueError(f'Normalizer {normalizer_name} is not supported!')


def get_continuous_imputer(imputer_name, random_state=42):
	"""
	Initializes an imputer for continuous variables

	Parameter
	---------
	imputer_name : str
		Name of the imputation method
	
	random_state : int
		Random seed to make the iterative imputer deterministic

	Returns
	-------
	imputer : object
		An imputer object
	"""

	if imputer_name == 'iterative':
		return IterativeImputer(max_iter=1000, random_state=random_state)
	elif imputer_name == 'mean':
		return SimpleImputer(missing_values=np.nan, strategy='mean')
	elif imputer_name == 'external':
		return SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
	else:
		raise ValueError(f'Imputer {imputer_name} is not supported.')


def get_categorical_imputer(imputer_name):
	"""
	Initializes an imputer for categorical variables

	Parameter
	---------
	imputer_name : str
		Name of the imputation method
	
	random_state : int
		Random seed to make the iterative imputer deterministic

	Returns
	-------
	imputer : object
		An imputer object
	"""

	if imputer_name == 'most_frequent':
		return SimpleImputer(missing_values=np.nan, strategy='most_frequent')
	elif imputer_name == 'external':
		return SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
	else:
		raise ValueError(f'Imputer {imputer_name} is not supported.')
