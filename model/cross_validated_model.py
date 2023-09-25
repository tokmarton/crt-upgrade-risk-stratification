import numpy as np
from model.feature_selection import FeatureSelectionCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, \
    recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from utils.utils import mean_confidence_interval

import warnings
warnings.filterwarnings('ignore')


class CrossValidatedModel:
    """
	A class that implements a machine learning model trained with grid search and nested cross-validation
	
	Attributes
	----------
	base_model :
		The machine learning model that will be trained
		
	cv : StratifiedKFold
		Cross-validation instance that will be used for both of the inner and outer loop of the nested cross-validation
		
	fit_models : 
		The machine learning models that have been fit with grid search on the different folds

	do_feature_selection: bool
		Controls whether the training will include feature selection

	feature_selection_params: dict
		Keyword arguments for the feature selection
		
	feature_selectors : list(FeatureSelectionCV)
		The FeatureSelectionCV instance corresponding to each of the models in fit_models
		
	best_features : set(str)
		The union of the best features selected by the feature selectors
		
	best_hyperparams : list(dict)
		The best hyperparameters selected by grid search
		
	param_grid : dict
		Dictionary of possible hyperparameters passed to grid search
		
	random_state : int
		Random seed to make the cross-validation deterministic
		
	n_jobs : int
		Number of jobs to run in parallelly
	"""

    def __init__(self, base_model, param_grid, scoring, random_state, weighted=False, do_feature_selection=False,
                 feature_selection_params=None, n_jobs=10, cv=5):
        self.base_model = base_model
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        self.fit_models = []
        self.feature_selectors = []
        self.best_features = set()
        self.best_hyperparams = []
        self.param_grid = param_grid
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.weighted = weighted
        self.do_feature_selection = do_feature_selection
        self.feature_selection_params = feature_selection_params

    def fit_gs(self, X, y):
        """
		Fits a model on each of the CV folds using grid search

		Parameters:
		-----------
		X : 2D array-like
			Input features

		y : 1D array-like
			Labels
		"""

        if self.do_feature_selection:
            self.__feature_selection_gs(X, y)
        else:
            self.__do_grid_search(X, y)

    def cross_validate(self, X, y, confidence=0.95):
        """
		Validates each fitted model on the corresponding test set
		
		Parameters
		----------
		X : 2D array-like
			Input features

		y : 1D array-like
			Labels
		
		confidence: float
			Confidence value used for calculating the confidence interval
			
		Returns
		-------
		Dictionary of performance metrics summarizing the results of the cross-validation
		"""

        if len(self.fit_models) == 0:
            self.fit_gs(X, y)

        # Calculate performance metrics on each fold
        accuracy = []
        balanced_accuracy = []
        precision_micro = []
        recall_micro = []
        f1_micro = []
        precision_macro = []
        recall_macro = []
        f1_macro = []
        roc_auc_micro = []
        roc_auc_macro = []
        for i, (train, test) in enumerate(self.cv.split(X, y)):
            if len(self.feature_selectors) > 0:
                X_test = self.feature_selectors[i].transform(X.iloc[test])
            else:
                X_test = X.iloc[test]

            # Calculate accuracy and balanced accuracy
            clf = self.fit_models[i]
            y_true = y.iloc[test]
            y_pred = clf.predict(X_test)

            accuracy.append(accuracy_score(y_true, y_pred))
            balanced_accuracy.append(balanced_accuracy_score(y_true, y_pred))

            # Calculate micro- and macro-average metrics (i.e., precision, recall, F1)
            precision_micro.append(precision_score(y_true, y_pred, average='micro'))
            recall_micro.append(recall_score(y_true, y_pred, average='micro'))
            f1_micro.append(f1_score(y_true, y_pred, average='micro'))

            precision_macro.append(precision_score(y_true, y_pred, average='macro'))
            recall_macro.append(recall_score(y_true, y_pred, average='macro'))
            f1_macro.append(f1_score(y_true, y_pred, average='macro'))

            # Calculate micro- and macro-average ROC AUC values
            y_proba = clf.predict_proba(X_test)
            roc_auc_micro.append(roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro'))
            roc_auc_macro.append(roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'))

        # Calculate confidence interval for each metric
        accuracy_mean, accuracy_lower, accuracy_upper = \
            mean_confidence_interval(accuracy, confidence)
        balanced_accuracy_mean, balanced_accuracy_lower, balanced_accuracy_upper = \
            mean_confidence_interval(balanced_accuracy, confidence)
        precision_micro_mean, precision_micro_lower, precision_micro_upper = \
            mean_confidence_interval(precision_micro, confidence)
        precision_macro_mean, precision_macro_lower, precision_macro_upper = \
            mean_confidence_interval(precision_macro, confidence)
        recall_micro_mean, recall_micro_lower, recall_micro_upper = \
            mean_confidence_interval(recall_micro, confidence)
        recall_macro_mean, recall_macro_lower, recall_macro_upper = \
            mean_confidence_interval(recall_macro, confidence)
        f1_micro_mean, f1_micro_lower, f1_micro_upper = \
            mean_confidence_interval(f1_micro, confidence)
        f1_macro_mean, f1_macro_lower, f1_macro_upper = \
            mean_confidence_interval(f1_macro, confidence)
        roc_auc_micro_mean, roc_auc_micro_lower, roc_auc_micro_upper = \
            mean_confidence_interval(roc_auc_micro, confidence)
        roc_auc_macro_mean, roc_auc_macro_lower, roc_auc_macro_upper = \
            mean_confidence_interval(roc_auc_macro, confidence)

        # Summarize results
        def mean_ci(mean_value, lower_limit, upper_limit):
            ci = str(round(lower_limit, 3)) + " - " + str(round(upper_limit, 3))
            return str(round(mean_value, 3)) + " [" + ci + "]"

        results = {'accuracy': mean_ci(accuracy_mean, accuracy_lower, accuracy_upper),
                   'balanced_accuracy':
                       mean_ci(balanced_accuracy_mean, balanced_accuracy_lower, balanced_accuracy_upper),
                   'precision_micro': mean_ci(precision_micro_mean, precision_micro_lower, precision_micro_upper),
                   'precision_macro': mean_ci(precision_macro_mean, precision_macro_lower, precision_macro_upper),
                   'recall_micro': mean_ci(recall_micro_mean, recall_micro_lower, recall_micro_upper),
                   'recall_macro': mean_ci(recall_macro_mean, recall_macro_lower, recall_macro_upper),
                   'f1_micro': mean_ci(f1_micro_mean, f1_micro_lower, f1_micro_upper),
                   'f1_macro': mean_ci(f1_macro_mean, f1_macro_lower, f1_macro_upper),
                   'roc_auc_micro': mean_ci(roc_auc_micro_mean, roc_auc_micro_lower, roc_auc_micro_upper),
                   'roc_auc_macro': mean_ci(roc_auc_macro_mean, roc_auc_macro_lower, roc_auc_macro_upper)}

        return results

    def predict_proba(self, X):
        """
		Calculates the predicted probabilities using all the fitted models
		
		Parameters
		----------
		X : 2D array-like
			Array of features
			
		Returns
		-------
		Predicted probabilities (1D array-like)
		"""

        # Ensure that the model was trained
        assert len(self.fit_models) != 0, 'Model has not yet been trained!'

        # Predict probabilities using models trained on the different folds of the training dataset
        probas = []
        for i, clf in enumerate(self.fit_models):
            if len(self.feature_selectors) > 0:
                X_transformed = self.feature_selectors[i].transform(X)
            else:
                X_transformed = X
            probas.append(clf.predict_proba(X_transformed))

        return np.mean(probas, axis=0)

    def __feature_selection_gs(self, X, y):
        for i, (train, test) in enumerate(self.cv.split(X, y)):
            fs = FeatureSelectionCV(self.base_model,
                                    self.param_grid,
                                    scoring=self.scoring,
                                    cv=self.cv,
                                    n_jobs=self.n_jobs,
                                    **self.feature_selection_params)
            w_train = None
            if self.weighted:
                w_train = compute_sample_weight('balanced', y.iloc[train])

            fs.fit(X.iloc[train], y.iloc[train], classifier__sample_weight=w_train)
            print(f'Fold-{i + 1} - {len(fs.best_features_)} features: {fs.best_score_: .4f}')
            self.feature_selectors.append(fs)
            self.fit_models.append(fs.best_estimator_)
            self.best_hyperparams.append(fs.best_params_)
            self.best_features.update(fs.best_features_)

    def __do_grid_search(self, X, y):
        for i, (train, test) in enumerate(self.cv.split(X, y)):
            gs = GridSearchCV(self.base_model,
                              self.param_grid,
                              scoring=self.scoring,
                              cv=self.cv, error_score='raise',
                              n_jobs=self.n_jobs)

            w_train = None
            if self.weighted:
                w_train = compute_sample_weight('balanced', y.iloc[train])

            gs.fit(X.iloc[train], y.iloc[train], classifier__sample_weight=w_train)
            self.fit_models.append(gs.best_estimator_)
            print(f'Fold-{i + 1}: {gs.best_score_}')
