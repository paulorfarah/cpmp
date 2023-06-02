import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def resampling_results():
    datasets = ['commons-bcel', 'commons-csv', 'commons-io', 'easymock', 'Openfire', 'pdfbox', 'wro4j']
    # algs = ['DT', 'LogisticRegression', 'MLP', 'RandomForest']
    # models = ['model1', 'model2', 'model3']
    res = pd.DataFrame(columns=['resample', 'F1-Score(None)', 'Accuracy', 'Sensitivity', 'ROC AUC score'])
    for dataset in datasets:
        df = pd.read_csv('results/cpmp/' + dataset + '-results-tradicional-no-feature-selection-model1-3.csv')
        df['F1-Score(None)'] = df['F1-Score(None)'].values[0].split(' ', 1)[0].replace('[', '')
        res = pd.concat([res, df[['resample', 'F1-Score(None)', 'Accuracy', 'Sensitivity', 'ROC AUC score']]])
    res.columns = ['resample', 'F1-Score', 'Accuracy', 'Sensitivity', 'AUC']
    res.sort_values(by=['resample'])
    res.to_csv('results/resampling.csv', index=False)

def best_resampling_results():
    df = pd.read_csv('results/cpmp/best_auc.csv')

def best_scores():
    datasets = ['commons-bcel', 'commons-csv', 'commons-io', 'easymock', 'Openfire', 'pdfbox', 'wro4j']
    algs = ['DT', 'LogisticRegression', 'MLP', 'RandomForest']
    models = ['model1', 'model2', 'model3']
    scores = pd.DataFrame(columns=['AP', 'MS', 'F1', 'Acc', 'Sen', 'AUC', 'RS', 'F1', 'Acc', 'Sen', 'AUC', 'RS'
        , 'F1', 'Acc', 'Sen', 'AUC', 'RS', 'F1', 'Acc', 'Sen', 'AUC', 'RS'])
    for dataset in datasets:
        df = pd.read_csv('results/cpmp/' + dataset + '-results-tradicional-no-feature-selection-model1-3.csv')
        for model in models:
            row_res = [dataset, model]
            for alg in algs:
                try:
                    id = df[(df['Algoritm'] == alg) & (df['model'] == model)]['ROC AUC score'].idxmax()
                    row = df.iloc[[id]]
                    f1 = row['F1-Score(None)'].values[0].split(' ', 1)[0].replace('[', '')
                    acc = row['Accuracy'].values[0]
                    sen = row['Sensitivity'].values[0]
                    auc = row['ROC AUC score'].values[0]
                    res = row['resample'].values[0]
                    row_res += [f1, acc, sen, auc, res]
                except:
                    print('ERROR: ' + str(sys.exc_info()))
                    row_res += [0, 0, 0, 0, 0]
            scores.loc[len(scores)] = row_res
    # print(res.head())
    scores.to_csv('results/cpmp/best_auc.csv', index=False)


def plot_boxplot_best_auc():
    df = pd.read_csv('results/cpmp/best_auc.csv')
    fig, ax = plt.subplots()
    df2 = df.iloc[:, [2, 7, 12, 17]]
    df2.columns = ['DT', 'LR', 'MLP', 'RF']
    df_melted = pd.melt(df2)
    # print(df_melted.head())
    # df2 = df.iloc[:, [1, 3, 4]]

    sns.boxplot(x='variable', y='value', data=df_melted)
    box = sns.boxplot(x='variable', y='value', data=df_melted, ax=ax).set(xlabel='ML Algorithm', ylabel='F1-Score')
    # fig = box.get_figure()
    plt.savefig('results/charts/f1-boxplot.png')
    plt.close()

    fig, ax = plt.subplots()
    df2 = df.iloc[:, [3, 8, 13, 18]]
    df2.columns = ['DT', 'LR', 'MLP', 'RF']
    df_melted = pd.melt(df2)
    # print(df_melted.head())
    # df2 = df.iloc[:, [1, 3, 4]]

    sns.boxplot(x='variable', y='value', data=df_melted)
    box = sns.boxplot(x='variable', y='value', data=df_melted, ax=ax).set(xlabel='ML Algorithm', ylabel='Accuracy')
    # fig = box.get_figure()
    plt.savefig('results/charts/acc-boxplot.png')
    plt.close()

    fig, ax = plt.subplots()
    df2 = df.iloc[:, [5, 10, 15, 20]]
    df2.columns = ['DT', 'LR', 'MLP', 'RF']
    df_melted = pd.melt(df2)

    sns.boxplot(x='variable', y='value', data=df_melted)
    box = sns.boxplot(x='variable', y='value', data=df_melted, ax=ax).set(xlabel='ML Algorithm', ylabel='ROC AUC')
    plt.savefig('results/charts/auc-boxplot.png')
    plt.close()


def concatenate_datasets(col1, col2):
    datasets = ['commons-bcel', 'commons-csv', 'commons-io', 'easymock', 'Openfire', 'pdfbox', 'wro4j']
    algs = ['DT', 'LogisticRegression', 'MLP', 'RandomForest']
    models = ['model1', 'model2', 'model3']
    scores = pd.DataFrame(columns=['AP', 'MS', 'F1', 'Acc', 'Sen', 'AUC', 'RS', 'F1', 'Acc', 'Sen', 'AUC', 'RS'
        , 'F1', 'Acc', 'Sen', 'AUC', 'RS', 'F1', 'Acc', 'Sen', 'AUC', 'RS'])

    df_res = pd.DataFrame()
    for dataset in datasets:
        df = pd.read_csv('results/cpmp/' + dataset + '-results-tradicional-no-feature-selection-model1-3.csv')
        df_res = pd.concat([df_res, df[[col1, col2]]])
    return df_res


def plot_boxplot_all():
    # cols: Dataset,Algoritm,window,model,resample,F1-Score(micro),F1-Score(macro),F1-Score(weighted),F1-Score(None),
    # Accuracy,Sensitivity,Specificity,ROC AUC score,Confusion matrix
    cols1 = ['Dataset', 'Algoritm', 'model', 'resample']
    cols2 = ['F1-Score(None)', 'Accuracy', 'Sensitivity', 'Specificity', 'ROC AUC score']
    for col1 in cols1:
        for col2 in cols2:
            df = concatenate_datasets(col1, col2)
            # df.loc[df[col1] == 'LogistRegression', df[col1]] = 'LogisticRegression'
            if col1 == 'Algoritm':
                df['Algoritm'] = df['Algoritm'].replace(['LogistRegression'], 'LogisticRegression')
            if col2 == 'F1-Score(None)':
                df['F1-Score(None)'] = df['F1-Score(None)'].str.split(' ').str[0]
                df['F1-Score(None)'] = df['F1-Score(None)'].str.replace('[', '')
                df['F1-Score(None)'] = df['F1-Score(None)'].astype(float)
            print(df.head())
            fig, ax = plt.subplots()
            # sns.boxplot(x='variable', y='value', data=df)
            box = sns.boxplot(x=col1, y=col2, data=df, ax=ax).set(xlabel=col1, ylabel=col2)
            # fig = box.get_figure()
            plt.xticks(rotation=15)
            plt.savefig('results/charts/' + col1 + '-' + col2 + '-boxplot.png')
            plt.close()

def plot_metrics():
    print('todo')

if __name__ == '__main__':
    # best_scores()
    # plot_boxplot_best_auc()
    # plot_boxplot_all()
    resampling_results()
