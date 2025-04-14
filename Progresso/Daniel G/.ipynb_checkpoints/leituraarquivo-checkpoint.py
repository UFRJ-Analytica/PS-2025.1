import pandas as pd

df = pd.read_csv('entrada.csv')
print(df[(df['Time 1'] == 'Livingston') & (df['Time 2'] == 'Motherwell')])