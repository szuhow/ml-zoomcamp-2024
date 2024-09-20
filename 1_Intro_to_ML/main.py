import pandas as pd

# print(pd.__version__)

data = pd.read_csv('laptops.csv')
# print(len(data))
#  print number of types of laptops
print(len(data['Brand'].value_counts()))
print(data.isnull().sum())