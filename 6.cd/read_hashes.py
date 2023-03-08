import pandas as pd

file = 'commons-bcel-results.csv'

df = pd.read_csv(file)
df.columns = ['project_name', 'commit_hash', 'previous_commit', 'file_name', 'previous_file', ]
print(df.columns)