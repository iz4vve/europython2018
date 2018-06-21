    @staticmethod
    def _calculate_importance(model, names=None):
        if hasattr(model, "feature_importances_"):
            return [(n, i) for n, i in enumerate(model.feature_importances_)]
        if hasattr(model, "variances_"):
            return [(n, i) for n, i in enumerate(model.variances_)]
        if hasattr(model, "scores_"):
            return [(n, i) for n, i in enumerate(model.scores_)]
        return list()


    def _get_importances(self):
        for key in self._alive_estimators:
            importances = self._calculate_importance(self._active_estimators[key], self.names)
            
            self._importances[key] = sorted(
                importances, key=operator.itemgetter(1), reverse=True
            )[:self.n_features]