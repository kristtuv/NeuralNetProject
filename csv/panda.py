import numpy as np

import pandas as pd

relu = pd.read_csv('relus10000.csv').head(5)
tanh = pd.read_csv('tanhs10000.csv').head(5)
sigmoid = pd.read_csv('sigmoids10000.csv').head(5)

df = pd.concat([relu, sigmoid, tanh])
df = df.drop(['Index', 'Critical error',  'Sample size', 'Test error', 'Train error'], axis=1)
df = df.fillna('None')
df = df.sort_values(by=['Critical accuracy'], ascending=False)
df = df.round(3)
df.to_csv('best_critcal.csv', sep='&', index=False)
print()
