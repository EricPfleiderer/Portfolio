import pandas as pd

data = pd.read_csv('data/eta_Bootis_weights.csv')

for x in range(data.shape[0]):
    data['weight'][x] = 1/data['weight'][x]

data.to_csv('data/eta_Bootis.csv')