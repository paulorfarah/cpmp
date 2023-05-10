import pandas as pd

if __name__ == '__main__':
    datasets = ['commons-bcel', 'commons-csv', 'commons-io', 'easymock', 'pdfbox', 'wro4j']
    algs = ['DT', 'LogisticRegression', 'MLP', 'RandomForest']
    models = ['model1', 'model2', 'model3']
    resample = ['NONE', 'ROS', 'SMOTE', 'ADA']
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
                    row_res += [0, 0, 0, 0, 0]
            scores.loc[len(scores)] = row_res
    # print(res.head())
    scores.to_csv('results/cpmp/best_auc.csv', index=False)