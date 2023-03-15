import pandas as pd

def join_evo(project_name, current_hash):
    print("evolutive")
    csv_path = '../3.evo/results/' + project_name + '-methods-results.csv'
    metrics = ["project", "commit", "commitprevious", "className", "BOC", "TACH", "FCH", "LCH", "CHO", "FRCH",
                      "CHD", "WCD", "WFR", "ATAF", "LCA", "LCD", "CSB", "CSBS", "ACDF"]

    df = pd.read_csv(csv_path, usecols=metrics, sep=',', engine='python', index_col=False)
    df = df[(df['commit'] == current_hash)]
    print(df.shape[0])
    return df
