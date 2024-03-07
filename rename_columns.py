import pandas as pd

def rename_columns(dataframe, new_column_names):
    dataframe.columns = new_column_names
    return dataframe

df = pd.read_csv('dataset.csv')

column_names = ['Class Label']

for j in range(1,20):
    for i in range(66):
        column_names.append(f'X{"{:02d}".format(i)}{"{:02d}".format(j)}')
        column_names.append(f'Y{"{:02d}".format(i)}{"{:02d}".format(j)}')
        column_names.append(f'Z{"{:02d}".format(i)}{"{:02d}".format(j)}')

df = rename_columns(df, column_names)

df.to_csv('dataset.csv', index=False)