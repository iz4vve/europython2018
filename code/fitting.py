    def fit(self, X, y=None):

        number_of_estimators = len(self._active_estimators)

        for n, (key, model) in enumerate(self._active_estimators.items(), start=1):
            model.fit(X, y)

            if not hasattr(model, "score"):
                self._alive_estimators.add(key)
                continue

            # minimum train score
            train_score = model.score(X, y)
            if not _check_score(train_score, self._min_score):
                continue

            # minimum test score
            if self._test_split:
                test_score = model.score(X_test, y_test)
                if not _check_score(test_score, self._min_score):
                    continue
            # all is well
            self._alive_estimators.add(key)
