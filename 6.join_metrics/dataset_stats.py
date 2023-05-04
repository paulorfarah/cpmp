import pandas as pd
from matplotlib import pyplot as plt

if __name__ == "__main__":
    projects = ['commons-csv', 'easymock']
    for project_name in projects:
        df = pd.read_csv('../6.join_metrics/results/' + project_name + '-perf-diff-all.csv')
    #     releases = df['release'].unique()
    #     res = df.groupby(['release', 'commit_hash', 'class']).count()
    #     print(res)
        # for release in releases:
        #
        print(len(df['release'].unique()))
        print(len(df['commit_hash'].unique()))
        print(len(df['class'].unique()))
        print(len(df['method'].unique()))
        print(len(df['method']))
        print(len(df['perf_change']))
        df_perf = df.groupby(['perf_change']).size()
        print(df_perf)