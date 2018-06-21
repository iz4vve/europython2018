    def fit(self, X, y=None):

        number_of_estimators = len(self._active_estimators)

        for n, (key, model) in enumerate(self._active_estimators.items(), start=1):
            model.fit(X, y)

            self._alive_estimators.add(key)
