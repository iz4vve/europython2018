#!/usr/bin/env python

import pandas as pd
from sklearn import datasets
# info on dataset available at: https://goo.gl/U2Uwz2
data = datasets.load_breast_cancer(return_X_y=False)

# START OMIT
df = pd.DataFrame(data['data'], columns=data['feature_names'])
print(f"Dataset contains {df.shape[0]} rows x {df.shape[1]} columns")
# END OMIT
df.columns = [f"X{i:02d}" for i in range(len(data['feature_names']))]

print("Column names:")
print(list(df)) 