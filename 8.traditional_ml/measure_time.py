import csv
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix, accuracy_score)
from sklearn.model_selection import StratifiedKFold as kf
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier


def generateStandardTimeSeriesStructure(all_releases_df, ws, featureList):
    print("Generating a new dataframe without containing the last release...")
    df = all_releases_df[all_releases_df['release'] != all_releases_df['release'].max()]
    print("... DONE!")

    df.drop(columns=["project", "commit", "TOTAL_CHANGES", "release", "will_change"])

    print("checking class larger than window size...")

    window_size = ws

    class_names_list = df['class'].unique().tolist()
    classes_to_drop_list = list()
    for class_name in class_names_list:
        if len(df[df['class'] == class_name].iloc[::-1]) <= window_size:
            for drop_class in df.index[df['class'] == class_name].tolist():
                classes_to_drop_list.append(drop_class)

    df = df.drop(classes_to_drop_list, axis=0)
    df = df.iloc[::-1]

    print("DONE")

    print("Setting the features...")
    class_names_list = df['class'].unique().tolist()
    features_list = featureList
    print("DONE")

    timeseries_list = list()
    timeseries_labels = list()
    for class_name in class_names_list:
        class_sequence = df[df['class'] == class_name].reset_index()
        for row in range(len(class_sequence) - 1):
            window = list()
            # print('row: ', row)
            if row + window_size < len(class_sequence) + 1:
                for i in range(window_size):
                    # print(row+i)
                    window.extend(class_sequence.loc[row + i, features_list].values.astype(np.float64))
                timeseries_labels.append(class_sequence.loc[row + i, 'will_change'])
                timeseries_list.append(window)

    timeseries_X = np.array(timeseries_list)
    timeseries_X = timeseries_X[:, ~np.isnan(timeseries_X).any(axis=0)]
    timeseries_labels = np.array(timeseries_labels).astype(np.bool)
    # np.savetxt("results/test.csv",timeseries_X, delimiter=",")

    return timeseries_X, timeseries_labels


def get_scores(y_test, y_pred, dataset, algorithm, rs, model, ws):
    scores = []
    scores.append(dataset)
    scores.append(algorithm)
    scores.append(ws)
    scores.append(model)
    scores.append(rs)

    scores.append(f1_score(y_test, y_pred, average='micro'))
    print("F1-Score(micro): " + str(scores[-1]))

    scores.append(f1_score(y_test, y_pred, average='macro'))
    print("F1-Score(macro): " + str(scores[-1]))

    scores.append(f1_score(y_test, y_pred, average='weighted'))
    print("F1-Score(weighted): " + str(scores[-1]))

    scores.append(f1_score(y_test, y_pred, average=None))
    print("F1-Score(None): " + str(scores[-1]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # ACC
    scores.append(accuracy_score(y_test, y_pred, normalize=True))
    print("Accuracy: " + str(scores[-1]))

    # precision
    precision = tp/ (tp + fp)
    scores.append(precision)

    # Sensitivity
    sensitivity = tp / (tp + fn)
    scores.append(sensitivity)
    print("Sensitivity: " + str(scores[-1]))

    # Specificity
    specificity = tn / (tn + fp)
    scores.append(specificity)
    print("Specificity: " + str(scores[-1]))

    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    print("Confusion Matrix: [" + str(cnf_matrix[0][0]) + ", " + str(round(cnf_matrix[1][1], 2)) + "]")
    plot_confusion_matrix(cnf_matrix, dataset)

    # ROC_AUC
    scores.append(roc_auc_score(y_test, y_pred))
    print("ROC AUC score: " + str(scores[-1]))

    scores.append([tn, fp, fn, tp])
    # head = ['Dataset', 'Algoritm', 'window', 'model', 'resample', 'F1-Score(micro)', 'F1-Score(macro)',
    #         'F1-Score(weighted)', 'F1-Score(None)', 'Accuracy', 'precision', 'Sensitivity', 'Specificity', 'ROC AUC score',
    #         'Confusion matrix']

    # if not os.path.exists('results/cpmp/' + dataset + '-results-tradicional-no-feature-selection-model1-3.csv'):
    #     f = open("results/cpmp/" + dataset + "-results-tradicional-no-feature-selection-model1-3.csv", "a")
    #     writer = csv.writer(f)
    #     writer.writerow(head)
    #     f.close()

    # f = open("results/cpmp/" + dataset + "-results-tradicional-no-feature-selection-model1-3.csv", "a")
    # writer = csv.writer(f)
    # writer.writerow(scores)
    # f.close()

    return scores


def plot_confusion_matrix(cm, dataset,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #    print("Normalized confusion matrix")
    # else:
    #    print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('results/cpmp/cf-' + dataset + '.png')
    plt.close()


def plot_confusion_matrixes(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # plt.subplots(1,2,figsize=(20,4))
    # plt.subplot(1,2,1)
    # plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plt.subplot(1,2,2)
    plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')

    plt.tight_layout()
    plt.show()


def LogisticRegr_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("\nLOGISTIC REGRESSION")
    cv_score = []
    i = 1
    # print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        # print('{} of KFold {}'.format(i, kf.n_splits))
        xtr_LR, xvl_LR = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_LR, yvl_LR = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        # model
        lr = LogisticRegression(solver='lbfgs', random_state=42, class_weight='balanced', n_jobs=-1)
        lr.fit(xtr_LR, ytr_LR.values.ravel())
        score = roc_auc_score(yvl_LR, lr.predict(xvl_LR))
        # print('ROC AUC score:', score)
        cv_score.append(score)
        i += 1

    # print('\nCROSS VALIDANTION SUMMARY:')
    # print('Mean: ' + str(np.mean(cv_score)))
    # print('Std deviation: ' + str(np.std(cv_score)))

    # print("\nTEST SET:")
    get_scores(Ytest, lr.predict(Xtest), dataset, "LogisticRegression", rs, model, ws)


def RandomForest_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("RANDOM FOREST")
    cv_score = []
    i = 1
    # print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        # print('{} of KFold {}'.format(i, kf.n_splits))
        xtr_RF, xvl_RF = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_RF, yvl_RF = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        # model
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, n_jobs=-1)
        rf.fit(xtr_RF, ytr_RF.values.ravel())
        score = roc_auc_score(yvl_RF, rf.predict(xvl_RF))
        # print('ROC AUC score:', score)
        cv_score.append(score)
        i += 1

    # print('\nCROSS VALIDANTION SUMMARY:')
    # print('Mean: ' + str(np.mean(cv_score)))
    # print('Std deviation: ' + str(np.std(cv_score)))

    # print("\nTEST SET:")
    get_scores(Ytest, rf.predict(Xtest), dataset, "RandomForest", rs, model, ws)


def NN_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("NEURAL NETWORK")
    cv_score = []
    i = 1
    # print("TRAIN AND VALIDATION SETS:")
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        # print('{} of KFold {}'.format(i, kf.n_splits))
        xtr_NN, xvl_NN = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_NN, yvl_NN = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        # model
        nn = MLPClassifier(random_state=42)
        # nn.fit(xtr_NN, ytr_NN.values.ravel())
        grid = GridSearchCV(nn, {}, n_jobs=-1,
                            verbose=0)
        grid.fit(xtr_NN, ytr_NN.values.ravel())
        score = roc_auc_score(yvl_NN, grid.predict(xvl_NN))
        # score = roc_auc_score(yvl_NN, nn.predict(xvl_NN))
        # print('ROC AUC score:', score)
        cv_score.append(score)
        i += 1

    # print('\nCROSS VALIDANTION SUMMARY:')
    # print('Mean: ' + str(np.mean(cv_score)))
    # print('Std deviation: ' + str(np.std(cv_score)))

    # print("\nTEST SET:")
    get_scores(Ytest, grid.predict(Xtest), dataset, "MLP", rs, model, ws)


def DecisionTree_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("\nDECISION TREE")
    cv_score = []
    i = 1
    # print("TRAIN AND VALIDATION SETS:")
    parameters = {'max_depth': range(1, 11)}
    for train_index, test_index in kf.split(Xtrain, Ytrain):
        # print('{} of KFold {}'.format(i, kf.n_splits))
        xtr_DT, xvl_DT = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
        ytr_DT, yvl_DT = Ytrain.iloc[train_index], Ytrain.iloc[test_index]

        # model
        dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        # dt.fit(xtr_DT, ytr_DT.values.ravel())

        grid = GridSearchCV(dt, {}, n_jobs=-1,
                            verbose=0)
        grid.fit(xtr_DT, ytr_DT.values.ravel())
        score = roc_auc_score(yvl_DT, grid.predict(xvl_DT))
        # score = roc_auc_score(yvl_DT, dt.predict(xvl_DT))
        # print('ROC AUC score:', score)
        cv_score.append(score)
        i += 1

    # print('\nCROSS VALIDANTION SUMMARY:')
    # print('Mean: ' + str(np.mean(cv_score)))
    # print('Std deviation: ' + str(np.std(cv_score)))

    # print("\nTEST SET:")
    get_scores(Ytest, grid.predict(Xtest), dataset, "DT", rs, model, ws)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    structural_metrics = ['line', 'cbo', 'cboModified', 'fanin', 'fanout', 'wmc',
                          'rfc', 'loc', 'returnsQty', 'variablesQty', 'parametersQty', 'methodsInvokedQty',
                          'methodsInvokedLocalQty', 'methodsInvokedIndirectLocalQty', 'loopQty', 'comparisonsQty',
                          'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
                          'mathOperationsQty', 'maxNestedBlocksQty', 'anonymousClassesQty', 'innerClassesQty',
                          'lambdasQty',
                          'uniqueWordsQty', 'modifiers', 'logStatementsQty', 'hasJavaDoc',
                          "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict",
                          "AvgEssential", "AvgLine", "AvgLineBlank", "AvgLineCode", "AvgLineComment", "CountClassBase",
                          "CountClassCoupled", "CountClassDerived", "CountDeclClass", "CountDeclClassMethod",
                          "CountDeclClassVariable", "CountDeclFile", "CountDeclFunction", "CountDeclInstanceMethod",
                          "CountDeclInstanceVariable", "CountDeclMethod", "CountDeclMethodAll",
                          "CountDeclMethodDefault",
                          "CountDeclMethodPrivate", "CountDeclMethodProtected", "CountDeclMethodPublic", "CountInput",
                          "CountLine", "CountLineBlank", "CountLineCode", "CountLineCodeDecl", "CountLineCodeExe",
                          "CountLineComment", "CountOutput", "CountPath", "CountSemicolon", "CountStmt",
                          "CountStmtDecl",
                          "CountStmtExe", "Cyclomatic", "CyclomaticModified", "CyclomaticStrict", "Essential",
                          "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential",
                          "MaxInheritanceTree", "MaxNesting", "PercentLackOfCohesion", "RatioCommentToCode",
                          "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential"]
    evolutionary_metrics = [
        'BOM', 'TACH', 'FCH', 'LCH', 'CHO', 'FRCH', 'CHD', 'WCH', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS',
        'ACDF',
    ]
    change_distiller_metrics = [
        "TOTAL_CHANGES"
        "will_change"
    ]

    model1 = structural_metrics
    model2 = structural_metrics + evolutionary_metrics
    model3 = evolutionary_metrics

    # datasets = ['commons-bcel','commons-io','junit4','pdfbox','wro4j']
    dataset_list = ['commons-bcel', 'commons-csv', 'commons-io', 'easymock', 'jgit', 'junit4', 'pdfbox', 'wro4j']
    datasets = ['Openfire']

    main_columns = [
        # ck
        'file', 'class', 'method', 'constructor', 'line', 'cbo', 'cboModified', 'fanin', 'fanout', 'wmc',
        'rfc', 'loc', 'returnsQty', 'variablesQty', 'parametersQty', 'methodsInvokedQty',
        'methodsInvokedLocalQty', 'methodsInvokedIndirectLocalQty', 'loopQty', 'comparisonsQty',
        'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
        'mathOperationsQty', 'maxNestedBlocksQty', 'anonymousClassesQty', 'innerClassesQty', 'lambdasQty',
        'uniqueWordsQty', 'modifiers', 'logStatementsQty', 'hasJavaDoc',

        # added
        'method_name', 'current_hash',
        # understand
        "Kind", "Name", "File", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict",
        "AvgEssential", "AvgLine", "AvgLineBlank", "AvgLineCode", "AvgLineComment", "CountClassBase",
        "CountClassCoupled", "CountClassDerived", "CountDeclClass", "CountDeclClassMethod",
        "CountDeclClassVariable", "CountDeclFile", "CountDeclFunction", "CountDeclInstanceMethod",
        "CountDeclInstanceVariable", "CountDeclMethod", "CountDeclMethodAll", "CountDeclMethodDefault",
        "CountDeclMethodPrivate", "CountDeclMethodProtected", "CountDeclMethodPublic", "CountInput",
        "CountLine", "CountLineBlank", "CountLineCode", "CountLineCodeDecl", "CountLineCodeExe",
        "CountLineComment", "CountOutput", "CountPath", "CountSemicolon", "CountStmt", "CountStmtDecl",
        "CountStmtExe", "Cyclomatic", "CyclomaticModified", "CyclomaticStrict", "Essential",
        "MaxCyclomatic", "MaxCyclomaticModified", "MaxCyclomaticStrict", "MaxEssential",
        "MaxInheritanceTree", "MaxNesting", "PercentLackOfCohesion", "RatioCommentToCode",
        "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential",

        # evometrics
        'project', 'commit',
        'commitprevious', 'release', 'file', 'method', 'BOM', 'TACH', 'FCH', 'LCH', 'CHO',
        'FRCH', 'CHD', 'WCH', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS', 'ACDF',
        # change distiller
        "PROJECT_NAME", "PREVIOUS_COMMIT", "CURRENT_COMMIT", "CLASS_CURRENTCOMMIT", "CLASS_PREVIOUSCOMMIT",
        "CLASS_CURRENTNAME", "CLASS_PREVIOUSNAME",
        "STATEMENT_DELETE", "STATEMENT_INSERT", "STATEMENT_ORDERING_CHANGE",
        "STATEMENT_PARENT_CHANGE", "STATEMENT_UPDATE", "TOTAL_STATEMENTLEVELCHANGES",
        "PARENT_CLASS_CHANGE", "PARENT_CLASS_DELETE", "PARENT_CLASS_INSERT", "CLASS_RENAMING",
        "TOTAL_CLASSDECLARATIONCHANGES",
        "RETURN_TYPE_CHANGE", "RETURN_TYPE_DELETE", "RETURN_TYPE_INSERT", "METHOD_RENAMING",
        "PARAMETER_DELETE", "PARAMETER_INSERT", "PARAMETER_ORDERING_CHANGE", "PARAMETER_RENAMING",
        "PARAMETER_TYPE_CHANGE", "TOTAL_METHODDECLARATIONSCHANGES",
        "ATTRIBUTE_RENAMING", "ATTRIBUTE_TYPE_CHANGE", "TOTAL_ATTRIBUTEDECLARATIONCHANGES",
        "ADDING_ATTRIBUTE_MODIFIABILITY", "REMOVING_ATTRIBUTE_MODIFIABILITY",
        "REMOVING_CLASS_DERIVABILITY", "REMOVING_METHOD_OVERRIDABILITY",
        "ADDING_CLASS_DERIVABILITY", "ADDING_CLASS_DERIVABILITY", "ADDING_METHOD_OVERRIDABILITY",
        "TOTAL_DECLARATIONPARTCHANGES", "TOTAL_CHANGES", "will_change"
    ]

    # resamples= ['NONE','RUS','ENN','TL','ROS','SMOTE','ADA']
    # resamples= ['RUS','ENN','TL','ROS','SMOTE','ADA']
    # resamples = ['NONE', 'ROS', 'SMOTE', 'ADA']
    #
    # resamples = ['RUS', 'ENN', 'TL']
    resamples = ['ROS']
    windowsize = [0]
    # models = [{'key': 'model1', 'value': model1}, {'key': 'model2', 'value': model2},
    #           {'key': 'model3', 'value': model3}]
    models = [{'key': 'model2', 'value': model2}]
    # models = [{'key':'model4', 'value': model4}, {'key': 'model6', 'value': model6}, {'key': 'model7', 'value': model7}]
    for dataset in datasets:
        for ws in windowsize:
            for rs in resamples:
                for model in models:
                    if dataset == 'all':
                        dfs = []
                        for ds_name in dataset_list:
                            dfs.append(pd.read_csv('../6.join_metrics/results/' + ds_name + '-all-releases.csv'))
                        all_releases_df = pd.concat(dfs)
                    else:
                        all_releases_df = pd.read_csv('../6.join_metrics/results/' + dataset + '-all-releases.csv')

                    all_releases_df = all_releases_df.fillna(0)
                    all_releases_df.columns = main_columns
                    # x_raw = all_releases_df[model.get('value')]
                    # y_raw = all_releases_df['will_change']
                    # Feature selection
                    # X_new = SelectPercentile(chi2, percentile=50).fit(x_raw, y_raw)
                    #  X_new = SelectFpr(chi2, alpha=0.05).fit(x_raw, y_raw)

                    #  mask = X_new.get_support()  # list of booleans
                    #   new_features = []  # The list of your K best features

                    # for bool, feature in zip(mask, model.get('value')):
                    #     if bool:
                    #          new_features.append(feature)
                    #   print(new_features)

                    # X, y = generateStandardTimeSeriesStructure(all_releases_df, ws, new_features)
                    # print("Filtering required columns into X features...")
                    X = all_releases_df[model.get('value')].copy()
                    print("... DONE!")

                    # print("Setting y column containing label of change-proneness...")
                    y = pd.DataFrame(all_releases_df.loc[:, 'will_change'])
                    # print("... DONE!")
                    # print("Declaring a dictionary to save results...")
                    results_dict = dict()
                    # print("... DONE!")

                    # print("Splitting dataset into train and test sets...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=42)
                    # print("General information:")
                    # print("X Train set:",
                    #       X_train.shape[0], "X Test set:", X_test.shape[0])
                    # print("y Train set:",
                    #       y_train.shape[0], "y Test set:", y_test.shape[0])
                    # print("... DONE!")

                    # print("Scaling features...")
                    scaler = MinMaxScaler()
                    X_train = pd.DataFrame(scaler.fit_transform(X_train))
                    X_test = pd.DataFrame(scaler.fit_transform(X_test))
                    # print("... DONE!")

                    # print("Setting stratified k-fold...")
                    k = 10
                    kf = StratifiedKFold(n_splits=k, shuffle=False)
                    # print("k =", k)
                    # print("... DONE!\n")
                    y_test = pd.DataFrame(y_test)
                    y_train = pd.DataFrame(y_train)

                    # WITHOUT OVER OR UNDERSUMPLING
                    if rs == 'NONE':
                        RandomForest_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                        DecisionTree_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                        LogisticRegr_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                        NN_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # UNERSAMPLING RUS','ENN','TL'
                    if rs == 'RUS':
                        X_RUS, y_RUS = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train.values.ravel())
                        y_RUS = pd.DataFrame(y_RUS)

                        RandomForest_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        DecisionTree_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        LogisticRegr_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        NN_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                    if rs == 'ENN':
                        X_ENN, y_ENN = EditedNearestNeighbours().fit_resample(X_train, y_train.values.ravel())
                        y_ENN = df = pd.DataFrame(y_ENN)
                        RandomForest_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                        DecisionTree_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                        LogisticRegr_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                        NN_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                    if rs == 'TL':
                        X_TL, y_TL = TomekLinks(n_jobs=-1).fit_resample(X_train, y_train.values.ravel())
                        y_TL = df = pd.DataFrame(y_TL)
                        RandomForest_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                        DecisionTree_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                        LogisticRegr_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                        NN_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # OVERSAMPLING 'ROS','SMOTE','ADA'
                    if rs == 'ROS':
                        start_time = time.time()
                        ros = RandomOverSampler(random_state=42)
                        X_ROS, y_ROS = ros.fit_resample(X_train, y_train)
                        print("--- %sROS seconds ---" % (time.time() - start_time))

                        start_time = time.time()
                        DecisionTree_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        print("--- %sDT seconds ---" % (time.time() - start_time))

                        start_time = time.time()
                        RandomForest_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        print("--- %sRF seconds ---" % (time.time() - start_time))

                        start_time = time.time()
                        LogisticRegr_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        print("--- %sLR seconds ---" % (time.time() - start_time))

                        start_time = time.time()
                        NN_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        print("--- %sNN seconds ---" % (time.time() - start_time))
                    if rs == 'SMOTE':
                        sm = SMOTE(random_state=42)
                        X_SMO, y_SMO = sm.fit_resample(X_train, y_train)
                        RandomForest_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                        DecisionTree_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                        LogisticRegr_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                        NN_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                    if rs == 'ADA':
                        ada = ADASYN(random_state=42)
                        X_ADA, y_ADA = ada.fit_resample(X_train, y_train)
                        RandomForest_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
                        DecisionTree_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
                        LogisticRegr_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
                        NN_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
