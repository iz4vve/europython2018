"""
Collection of functions to perform feature selection.
Copyright (C) 2018 - Pietro Mascolo

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Pietro Mascolo
Email: iz4vve@gmail.com
"""
import time
# START IMPORT OMIT
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold
# END IMPORT OMIT

class EnsembleFeatureSelector(BaseEstimator):
    """
    EnsembleFeatureSelector builds an ensemble of weak models to perform
    feature selection.

    All models will compute what features are most important wrt a target
    variable. After training, all models get to vote for the features they
    believe are more relevant for the target.
    The overall feature importance will be given by the number of votes a
    feature obtains.
    """
    __regressors = {
        "RandomForestRegressor": RandomForestRegressor,
        "DecisionTreeRegressor": DecisionTreeRegressor
    }
    __classifiers = {
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier
    }
    __others = {
        "VarianceThreshold": VarianceThreshold,
    }

    __allowed_analyses = {
        "classification",
        "regression",
        "generic_regression",
        "generic_classification"
    }

    __estimators_by_type = {
        "classification": ["__classifiers"],
        "regression": ["__regressors"],
        "generic_classification": ["__classifiers", "__others"],
        "generic_regression": ["__regressors", "__others"]
    }

    def __init__(
        self, number_of_features=-1, analysis_type=None,
        params=dict(), min_score=0.7, test_split=0, allowed_discrepancy=0
    ):
        """
        Initialisation of the ensemble model. A type of analysis must be
        specified so that the appropriate models can be selected and
        instantiated.
        
        After scoring all features, only models with a high enough
        score will be considered and allowed to vote.

        :param number_of_features: maximum number of features to score
                                   -1 will score all features (DO NOT USE
                                   for high dimensional datasets)
        :param analysis_type: type of analysis to be performed
                              (classification/regression)
        :param params: dictionary of parameters for the models that will
                       be instantiated
        :param min_score: minimum training score to keep a model and allow
                          it to vote
        :param test_split: split ratio for train/test (only useful in 
                           conjunction with allowed_discrepancy)
        :param allowed_discrepancy: maximum discrepancy in score between
                                    train and test allowed. Models with higher
                                    discrepancy are discarded. Only used if
                                    test_split > 0
        """

        if analysis_type not in self.__allowed_analyses:
            raise RuntimeError(
                (
                    "Analysis type not supported."
                    f" Available anlyses: {self.__allowed_analyses}"
                )
            )
        if not 0 <= test_split < 1 or not 0 <= min_score < 1:
            raise RuntimeError("test_split and min_score must be in [0; 1[")
        if test_split > 0.5:
            raise RuntimeWarning(
                "WARNING: test_split > 0.5 might yield inaccurate results"
            )

        if not 0 <= allowed_discrepancy <= 0.5:
            raise RuntimeError(
                "allowed_discrepancy must be in the interval [0; 5]"
            )
        if allowed_discrepancy and not test_split:
            raise RuntimeWarning(
                (
                    "WARNING: setting allowed_discrepancy without having set"
                    " test_split. This will have no effect"
                )
            )
        self._min_score = min_score
        self._allowed_discrepancy = allowed_discrepancy
        self._test_split = test_split

        self.analysis_type = analysis_type

        # self._estimators = dict()
        self._importances = dict()
        self.n_features = number_of_features

        self._active_estimators = dict()
        self._setup_models(params)

    def _setup_models(self, params):
        """
        _setup_models registers appropriate models for the requested
        analysis_type.

        :param params: dictionary of parameters for models
        """
        estimators = self.__estimators_by_type.get(self.analysis_type)
        print(f"Setting up models: {estimators}")
        for item in estimators:
            for label, est in self.__class__.__dict__[
                f"_{self.__class__.__name__}{item}"
            ].items():
                estimator_kwargs = params.get(label, dict())
                print(f"Adding {label} to active estimators")
                self._active_estimators[label] = est(**estimator_kwargs)

    def fit(self, X, y):
        pass
    
    def _calculate_importance(self):
        pass
