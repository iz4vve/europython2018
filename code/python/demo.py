#!/usr/bin/env python
import pandas as pd
from sklearn import datasets
# info on dataset available at: https://goo.gl/U2Uwz2

data = datasets.load_breast_cancer(return_X_y=False)
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
# TODO - introduce weights for votes

import collections
import itertools
import operator

from sklearn.base import BaseEstimator

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest,\
    SelectFwe, SelectFdr, SelectFpr

df = pd.DataFrame(data['data'])
target = pd.Series(data['target'])
print(f"Dataset contains {df.shape[0]} rows x {df.shape[1]} columns")

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
        "DecisionTreeRegressor": DecisionTreeRegressor,
    }
    __classifiers = {
        "RandomForestClassifier": RandomForestClassifier,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }
    __others = {
        "VarianceThreshold": VarianceThreshold,
    }

    __requires_scaling = {
        "VarianceThreshold"
    }

    __allowed_analyses = {
        "generic",
        "classification",
        "regression",
        "generic_regression",
        "generic_classification"
    }

    __estimators_by_type = {
        "generic": ["__others"],
        "classification": ["__classifiers"],
        "regression": ["__regressors"],
        "generic_classification": ["__classifiers", "__others"],
        "generic_regression": ["__regressors", "__others"]
    }

    def __init__(
        self, number_of_features=10, analysis_type="generic",
        params=None, min_score=0.7, test_split=0., allowed_discrepancy=0.,
        names=None, verbose=True
    ):
        """
        Initialisation of the ensemble model. A type of analysis must be
        specified so that the appropriate models can be selected and
        instantiated.
        
        After scoring all features, only models with a high enough
        score will be considered and allowed to vote.

        :param number_of_features: maximum number of features models will select
        :param analysis_type: type of analysis to be performed
                              (classification/regression/generic).
                              This parameter determines the estimators that
                              will be used.
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
        :param names: feature names
        :param verbose: sets verbose behaviour
        """
        if params is None:
            params = dict()

        if analysis_type not in self.__allowed_analyses:
            raise RuntimeError(
                (
                    "Analysis type not supported."
                    f" Available analyses: {self.__allowed_analyses}"
                )
            )
        if not 0. <= test_split < 1. or not 0. <= min_score < 1.:
            raise RuntimeError("test_split and min_score must be in [0; 1[")
        if test_split > 0.5:
            raise RuntimeWarning(
                "WARNING: test_split > 0.5 might yield inaccurate results"
            )

        if not 0. <= allowed_discrepancy <= 0.5:
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
        self.names = names
        self.verbose = verbose
        self._min_score = min_score
        self._allowed_discrepancy = allowed_discrepancy
        self._test_split = test_split

        self.analysis_type = analysis_type

        self._importances = dict()
        self.n_features = number_of_features

        self._active_estimators = dict()
        # labels of estimators that match criteria
        self._alive_estimators = set()
        self._setup_models(params)

        self._is_fit = False

    def _setup_models(self, params):
        """
        _setup_models registers appropriate models for the requested
        analysis_type.

        :param params: dictionary of parameters for models
        """
        estimators = self.__estimators_by_type.get(self.analysis_type, list())
        if self.verbose:
            print(f"Setting up models: {estimators}")
        for item in estimators:
            for label, est in self.__class__.__dict__[
                f"_{self.__class__.__name__}{item}"
            ].items():
                estimator_kwargs = params.get(label, dict())
                if self.verbose:
                    print(f"Adding {label} to active estimators")
                self._active_estimators[label] = est(**estimator_kwargs)

    def fit(self, X, y=None):
        """
        Fits all models in _active_estimators
        :param X: train dataset
        :param y: train labels
        """
        X_test, y_test =None, None

        if self._test_split > 0:
            X, X_test, y, y_test = train_test_split(
                X, y, test_size=self._test_split
            )

        if y is None and self.analysis_type != "generic":
            raise RuntimeError("y must be set when analysis is not generic")

        number_of_estimators = len(self._active_estimators)

        for n, (key, model) in enumerate(
            self._active_estimators.items(), start=1
        ):
            if self.verbose:
                print(
                    (f"Training instance of {model.__class__}"
                     f" ({n}/{number_of_estimators})")
                )
            if model.__class__.__name__ in self.__requires_scaling:
                X = (X - X.mean()) / (X.max() - X.min())
            model.fit(X, y)

            if not hasattr(model, "score"):
                if self.verbose:
                    print(
                        (f"{model.__class__} has no attribute score."
                         " Skipping tests...")
                    )
                self._alive_estimators.add(key)
                continue

            # minimum train score
            train_score = model.score(X, y)
            if not _check_score(train_score, self._min_score):
                if self.verbose:

                    print(
                        (f"{model.__class__} has train score of {train_score}."
                         f" Threshold: {self._min_score}."
                         " Model not accurate enough. Skipping...")
                    )
                continue

            # minimum test score
            if self._test_split:
                test_score = model.score(X_test, y_test)
                if not _check_score(test_score, self._min_score):
                    if self.verbose:
                        print(
                            (f"{model.__class__} has test score of {test_score}."
                             f" Threshold: {self._min_score}."
                             " Model not accurate enough. Skipping...")
                        )
                    continue

                # maximum allowed discrepancy
                if not _check_discrepancy(
                        train_score, test_score, self._allowed_discrepancy
                ):
                    if self.verbose:
                        discrepancy = abs(train_score - test_score)
                        print(
                            (f"{model.__class__} has train-test discrepancy of" 
                             f" {discrepancy}. Threshold:"
                             f" {self._allowed_discrepancy}."
                             " Model not stable enough. Skipping...")
                        )
                    continue

            # all is well
            self._alive_estimators.add(key)
        if not self._alive_estimators:
            print(
                ("No model met analysis criteria."
                 " Impossible to provide feature importances")
            )
        else:
            self._is_fit = True
        if self.verbose:
            print(
                f"Training completed. {len(self._active_estimators)} trained"
            )

    @staticmethod
    def _calculate_importance(model, names=None):
        """
        Returns feature importance for a given model
        :param model: trained sklearn estimator
        :sets: feature importance or variances
        """
        if hasattr(model, "feature_importances_"):
            if names is not None:
                return list(zip(names, model.feature_importances_))
            return [(n, i) for n, i in enumerate(model.feature_importances_)]
        if hasattr(model, "variances_"):
            if names is not None:
                return list(zip(names, model.variances_))
            return [(n, i) for n, i in enumerate(model.variances_)]
        if hasattr(model, "scores_"):
            if names is not None:
                return list(zip(names, model.scores_))
            return [(n, i) for n, i in enumerate(model.scores_)]
        return list()

    def _get_importances(self):
        """
        Calculates importance for each model in _alive_estimators
        """
        for key in self._alive_estimators:
            importances = self._calculate_importance(
                self._active_estimators[key], self.names
            )
            self._importances[key] = sorted(
                importances, key=operator.itemgetter(1), reverse=True
            )[:self.n_features]

    def cast_votes(self, min_votes=0):
        """
        Makes the Ensemble reach a consensus on the most important features
        """
        if not self._is_fit:
            raise RuntimeError(
                "Ensemble has not been fitted on data. Cannot cast votes"
            )

        self._get_importances()

        votes = collections.Counter(
            feature for feature, _ in
            itertools.chain(*self._importances.values())
        )
        return collections.Counter({
            feature: vote for feature, vote in
            votes.items() if vote >= min_votes
        })


def _check_score(score, threshold):
    return score > threshold


def _check_discrepancy(train, test, threshold):
    return abs(train - test) < threshold

df.columns = columns=[f"X{i:02d}" for i in range(len(data['feature_names']))]


def pretty_print_votes(votes):
    pass

def pretty_print_importances(importances):
    pass

from numpy.random import seed
seed(12345)

EFS = EnsembleFeatureSelector(
    analysis_type="generic_classification",
    names=df.columns,
    number_of_features=5
)

EFS.fit(df, target)
votes = EFS.cast_votes()

pretty_print_votes(votes)
pretty_print_importances(EFS._importances)

from sklearn.linear_model import LogisticRegression

logres = LogisticRegression()
logres_important = LogisticRegression()


# START OMIT
seed(12345)
df_train, df_test, target_train, target_test = train_test_split(df, target)

logres.fit(df_train, target_train)


seed(12345)
df_train_i, df_test_i, target_train_i, target_test_i = train_test_split(
    df[list(votes.keys())], target
)

logres_important.fit(df_train_i, target_train_i)

# END OMIT
print("*" * 80)
print("Model scores:")
print(
    f"Model score using {df_train.shape[1]} features: {logres.score(df_test, target_test):.4f}"
)

print(
    (f"Model score using {df_train_i.shape[1]} features:"
     f" {logres_important.score(df_test_i, target_test_i):.4f}")
)