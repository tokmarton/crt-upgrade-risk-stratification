import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils.preprocess_utils import get_categorical_imputer, get_normalizer, get_continuous_imputer


class BioDataPreprocess:
    """
	A class that defines the preprocessing steps and creates the training pipeline
	
	Attributes
	----------
	data : pd.DataFrame
		The dataset used for training

	target_column : str
		The name of the column containing the labels

	base_model 
		The machine learning model used in the last step of the pipeline

    smote : bool
        Whether Synthetic Minority Oversampling Technique (SMOTE) should be applied to tackle class imbalance

	drop_threshold : float
		The threshold for the proportion of missing values above which the row or column will be dropped

	normalizer : str
		Name of the normalization method

	categorical_impute : str
		Name of the imputation method for categorical variables

	continuous_impute : str
		Name of the imputation method for continuous variables

	random_state : int
		Random seed to make the iterative imputer deterministic
	"""

    def __init__(self, data: pd.DataFrame,
                 target_column: str,
                 base_model,
                 smote: bool = False,
                 drop_threshold: int = 0.3,
                 normalizer: str = None,
                 categorical_impute: str = 'most_frequent',
                 continuous_impute: str = 'mean',
                 random_state=42):
        self.data = data
        self.target_column = target_column
        self.base_model = base_model
        self.smote = smote
        self.drop_threshold = drop_threshold
        self.normalizer = normalizer
        self.categorical_impute = categorical_impute
        self.continuous_impute = continuous_impute
        self.random_state = random_state

    def preprocess_and_create_pipeline(self):
        """
		Splits the data into input features and labels and creates the training pipeline
		
		Returns
		-------
		X : 2D array-like
			Input features

		y : 1D array-like
			Labels
			
		pipeline : Pipeline
			The training pipeline
		"""

        # Copy data
        X = self.data

        # Exclude patients and features with a proportion of missing values above the predefined threshold
        X = X.loc[X.isna().mean(axis=1) < self.drop_threshold, X.isna().mean(axis=0) < self.drop_threshold]

        # Set the label and remove it from the input features
        y = X[self.target_column]
        X = X.drop([self.target_column], axis=1)

        # Remove columns that are not used for training
        input_features = ['age', 'sex', 'crt_d', 'nyha', 'afib', 'htn', 'diabetes', 'etiology',
                          'mi', 'pci', 'cabg', 'creat', 'gfr', 'lvef', 'lvidd', 'lvids']
        columns_to_drop = np.setdiff1d(X.columns, np.array(input_features))
        X = X.drop(columns_to_drop, axis=1)

        # Remove patients who do not belong to any risk group or belong to multiple groups
        X = X[y > 0]
        y = y[y > 0]

        # Set classes to [0, 1, 2]
        y = y - 1

        # Indentify continuous and categorical features
        continuous_columns = [col for col in X if len(X[col].dropna().unique()) > 10]
        categorical_columns = [col for col in X if len(X[col].dropna().unique()) <= 10]

        # Initialize the column transformer
        col_transformer = ColumnTransformer(
            [
                ('categorical', self.__preprocess_categorical_columns(),
                 lambda df: [c for c in df.columns if c in categorical_columns]),
                ('continuous', self.__preprocess_continuous_columns(),
                 lambda df: [c for c in df.columns if c in continuous_columns]),
            ]
        )

        # Define the steps of preprocessing
        preproc_steps = [
            ('preprocessor', col_transformer),
            ('classifier', self.base_model)
        ]
        if self.smote:
            preproc_steps.insert(1, ('smote', SMOTE(random_state=self.random_state)))

        # Create the pipeline for training
        pipeline = Pipeline(steps=preproc_steps)

        return X, y, pipeline

    def __preprocess_continuous_columns(self, normalize=True):
        imputer = get_continuous_imputer(self.continuous_impute, self.random_state)
        normalizer = get_normalizer(self.normalizer) if normalize else get_normalizer('')
        return Pipeline(steps=[('imputer', imputer), ('normalizer', normalizer)])

    def __preprocess_categorical_columns(self):
        imputer = get_categorical_imputer(self.categorical_impute)
        return Pipeline(steps=[('imputer', imputer)])
