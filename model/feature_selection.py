from mrmr import mrmr_classif
from sklearn.model_selection import GridSearchCV


class FeatureSelectionCV:
    """
	A class that fits a model with hyperparameter optimization and feature selection using cross-validation
	Serves as the inner loop in the nested cross-validation
	
	Parameters
	----------
	estimator :
		The machine learning estimator that will be fitted
		
	hyper_params : dict
		Dictionary of possible hyperparameters passed to grid search
		
	cv : 
		Cross-validation instance that is used for grid search
		
	scoring : str
		Evaluation metric
		
	step : int
		The number of features to be removed in each step
		
	min_features : int
		The minimum number of selected features
	
	max_features : int
		The maximum number of selected features
		
	n_jobs : int
		Number of jobs to run in parallelly
	"""

    def __init__(self, estimator, hyper_params: dict, cv=None, scoring='balanced_accuracy', step=1, min_features=10,
                 max_features=20, score_threshold=0.95, n_jobs=10):
        self.best_params_ = None
        self.estimator = estimator
        self.hyper_params = hyper_params
        self.cv = cv
        self.scoring = scoring
        self.step = step
        self.score_threshold = score_threshold
        self.min_features = min_features
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.scores = dict()
        self.score_stds = None
        self.best_estimator_ = None
        self.best_features_ = None
        self.best_score_ = 0

    @staticmethod
    def calculate_feature_importances(X, y, K=None):
        """
		Calculates the feature importances using the mRMR algorithm

		Parameters
		----------
		X : 2D array-like
			Input features

		y : 1D array-like
			Labels

		K : int
			Number of features to be selected

		Returns
		-------
		Ranked list containing K selected features
		"""

        if K is None:
            K = X.shape[0]

        return mrmr_classif(X, y, K=K)

    def fit(self, X, y, **fit_params):
        """
		Fits the model using grid search and feature selection
		
		Parameters
		----------
		X : 2D array-like
			Input features

		y : 1D array-like
			Labels
		
		fit_params : dict
			Additional fitting parameters
		"""

        current_features = self.calculate_feature_importances(X, y)
        gs = GridSearchCV(self.estimator, self.hyper_params, scoring=self.scoring, cv=self.cv, error_score='raise',
                          n_jobs=self.n_jobs)
        gs.fit(X, y, **fit_params)
        initial_score = gs.best_score_
        initial_std = gs.cv_results_['std_test_score'][gs.best_index_]
        print(f'{X.shape[1]} features:{initial_score: .4f} +/-{initial_std: .4f}')

        if self.max_features > X.shape[1]:
            self.max_features = X.shape[1]

        self.best_params_ = gs.best_params_
        self.best_score_ = gs.best_score_
        self.best_estimator_ = gs.best_estimator_
        self.best_features_ = X.columns

        self.scores[X.shape[1]] = initial_score
        self.score_stds = [gs.cv_results_['std_test_score'][gs.best_index_]]

        X_current = X.copy()
        for i in range(self.max_features - self.step, self.min_features - 1, -self.step):
            current_features = current_features[:i]

            X_current = self.__transform(X_current, current_features)
            gs.fit(X_current, y, **fit_params)
            current_score = gs.best_score_
            current_std = gs.cv_results_['std_test_score'][gs.best_index_]
            print(f'{X_current.shape[1]} features:{current_score: .4f} +/-{current_std: .4f}')

            if current_score > self.best_score_ and current_score > initial_score * self.score_threshold:
                self.best_params_ = gs.best_params_
                self.best_score_ = gs.best_score_
                self.best_estimator_ = gs.best_estimator_
                self.best_features_ = current_features

            self.scores[i] = current_score
            self.score_stds.append(current_std)

        if self.best_score_ == 0:
            self.best_score_ = initial_score

    def transform(self, X):
        """
		Transforms the input array to an array including the best selected features only
		
		Parameters
		----------
		X : 2D array-like
			Input features
		
		Returns
		-------
		Transformed array
		"""

        return self.__transform(X, self.best_features_)

    def fit_transform(self, X, y, **fit_params):
        """
		Fits the model and transforms the input array
		
		Parameters
		----------
		X : 2D array-like
			Input features

		y : 1D array-like
			Labels
		
		fit_params : dict
			Additional fitting parameters

		Returns
		-------
		Fitted and transformed array
		"""

        self.fit(X, y, **fit_params)
        return self.transform(X)

    @staticmethod
    def __transform(X, features):
        return X[features]
