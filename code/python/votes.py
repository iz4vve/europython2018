    def cast_votes(self, min_votes=0):
        if not self._is_fit:
            raise RuntimeError("Ensemble has not been fit on data. Cannot cast votes")

        self._get_importances()

        votes = collections.Counter(
            feature for feature, _ in itertools.chain(*self._importances.values())
        )
        return collections.Counter(
            {feature: vote for feature, vote in votes.items() if vote >= min_votes}
        )