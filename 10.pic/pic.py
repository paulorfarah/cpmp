import csv
import os
from pathlib import Path

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, multilabel_confusion_matrix, \
    precision_recall_curve, PrecisionRecallDisplay, average_precision_score, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# def plot_precision_recall_curve(precision, recall, average_precision, n_classes):
#     # global y, i
#     # fig, ax = plt.subplots(figsize=(8, 5))
#     from itertools import cycle
#     # setup plot details
#     colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
#     plt.figure(figsize=(7, 8))
#     f_scores = np.linspace(0.2, 0.8, num=4)
#     lines = []
#     labels = []
#     for f_score in f_scores:
#         x = np.linspace(0.01, 1)
#         y = f_score * x / (2 * x - f_score)
#         l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
#         plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
#         lines.append(l)
#     labels.append('iso-f1 curves')
#     l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
#     lines.append(l)
#     labels.append('micro-average Precision-recall (area = {0:0.2f})'
#                   ''.format(average_precision["micro"]))
#     for i, color in zip(range(n_classes), colors):
#         l, = plt.plot(recall[i], precision[i], color=color, lw=2)
#         lines.append(l)
#         labels.append('Precision-recall for class {0} (area = {1:0.2f})'
#                       ''.format(i, average_precision[i]))
#     fig = plt.gcf()
#     fig.subplots_adjust(bottom=0.25)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Extension of Precision-Recall curve to multi-class')
#     plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
#     plt.show()

# def get_scores(y_test, y_pred, dataset, algorithm, rs, model, ws, params=[]):
#     scores = []
#     scores.append(dataset)
#     scores.append(algorithm)
#     scores.append(ws)
#     scores.append(model)
#     scores.append(rs)
#
#     scores.append(f1_score(y_test, y_pred, average='micro'))
#     print("F1-Score(micro): " + str(scores[-1]))
#
#     scores.append(f1_score(y_test, y_pred, average='macro'))
#     print("F1-Score(macro): " + str(scores[-1]))
#
#     scores.append(f1_score(y_test, y_pred, average='weighted'))
#     print("F1-Score(weighted): " + str(scores[-1]))
#
#     scores.append(f1_score(y_test, y_pred, average=None))
#     print("F1-Score(None): " + str(scores[-1]))
#
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#
#     # ACC
#     scores.append(accuracy_score(y_test, y_pred, normalize=True))
#     print("Accuracy: " + str(scores[-1]))
#
#     # precision
#     precision = tp / (tp + fp)
#     scores.append(precision)
#     print("Precision: " + str(scores[-1]))
#
#     # Sensitivity / recall
#     sensitivity = tp / (tp + fn)
#     scores.append(sensitivity)
#     print("Sensitivity: " + str(scores[-1]))
#
#     # Specificity
#     specificity = tn / (tn + fp)
#     scores.append(specificity)
#     print("Specificity: " + str(scores[-1]))
#
#     # Confusion Matrix
#     cnf_matrix = confusion_matrix(y_test, y_pred)
#     cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
#     print("Confusion Matrix: [" + str(cnf_matrix[0][0]) + ", " + str(round(cnf_matrix[1][1], 2)) + "]")
#     plot_confusion_matrix(cnf_matrix, dataset)
#
#     # ROC_AUC
#     scores.append(roc_auc_score(y_test, y_pred))
#     print("ROC AUC score: " + str(scores[-1]))
#
#     scores.append([tn, fp, fn, tp])
#     head = ['Dataset', 'Algoritm', 'window', 'model', 'resample', 'F1-Score(micro)', 'F1-Score(macro)',
#             'F1-Score(weighted)', 'F1-Score(None)', 'Accuracy', 'precision', 'Sensitivity', 'Specificity',
#             'ROC AUC score',
#             'Confusion matrix']
#
#     if not os.path.exists('results/cpmp/' + dataset + '-results-hist-model1-3.csv'):
#         f = open("results/cpmp/" + dataset + "-results-hist-model1-3.csv", "a")
#         writer = csv.writer(f)
#         writer.writerow(head)
#         f.close()
#
#     f = open("results/cpmp/" + dataset + "-results-hist-model1-3.csv", "a")
#     writer = csv.writer(f)
#     writer.writerow(scores)
#     f.close()
#
#     if params:
#         params = scores.append(params)
#         f = open("results/cpmp/" + dataset + "-hist-params.csv", "a")
#         writer = csv.writer(f)
#         writer.writerow(params)
#         f.close()
#
#     return scores

# def multilabel_results(y_test, y_pred, algorithm, dataset, rs, model, ws):
#
#     # X, y = make_multilabel_classification(n_features=20, n_classes=2)
#     # Train/test sets
#     # X_train, X_test, y_train, y_test = train_test_split(
#     #     X, y, test_size=0.3, random_state=1121218
#     # )
#     #
#     # # Fit/predict
#     # etc = ExtraTreesClassifier()
#     # _ = etc.fit(X_train, y_train)
#     # y_pred = etc.predict(X_test)
#
#     # parameters = {
#     #         'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=200)],
#     #         'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
#     #     }
#     # c = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100, n_jobs=-1)
#
#     # Plot confusion matrix
#     fig, ax = plt.subplots(figsize=(8, 5))
#     # y_test.values.argmax(axis=1), predictions.argmax(axis=1))
#
#     scores = []
#     scores.append(dataset)
#     scores.append(algorithm)
#     scores.append(ws)
#     scores.append(model)
#     scores.append(rs)
#
#     scores.append(f1_score(y_test, y_pred, average='micro'))
#     print("F1-Score(micro): " + str(scores[-1]))
#
#     mcm = multilabel_confusion_matrix(y_test, y_pred)
#     print(mcm)
#
#     # precision, recall, _ = precision_recall_curve(y_test, y_pred)
#     # disp = PrecisionRecallDisplay(precision=precision, recall=recall)
#     # disp.plot()
#     # plt.show()
#
#     precision = dict()
#     recall = dict()
#     treshold = dict()
#     average_precision = dict()
#
#
#     n_classes = y_test.shape[1]
#
#     for i in range(n_classes):
#         precision[i], recall[i], treshold[i] = precision_recall_curve(y_test[:, i], y_pred[:, i])
#         average_precision[i] =average_precision_score(y_test[:, i], y_pred[:, i])
#
#     print("average_precision {} ".format(average_precision))
#
#     plt.figure()
#
#     for i in range(n_classes):
#       plt.step(recall[i], precision[i], where='post')
#       plt.xlabel('Recall')
#       plt.ylabel('Precision')
#       plt.ylim([0.0, 1.05])
#       plt.xlim([0.0, 1.0])
#       plt.title(
#         'Average precision score for class {}: AP={:.3f}'.format(i, average_precision[i]))
#       plt.show()
#
#       # A "micro-average": quantifying score on all classes jointly
#       precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
#                                                                       y_pred.ravel())
#       average_precision["None"] = average_precision_score(y_test, y_pred, average=None)
#       average_precision["Void"] = average_precision_score(y_test, y_pred)
#       average_precision["micro"] = average_precision_score(y_test, y_pred, average="micro")
#       average_precision["macro"] = average_precision_score(y_test, y_pred, average="macro")
#       average_precision["samples"] = average_precision_score(y_test, y_pred, average="samples")
#       average_precision["weighted"] = average_precision_score(y_test, y_pred, average="weighted")
#
#       # print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
#
#       print(average_precision)
#       print(precision)
#
#       plot_precision_recall_curve(precision, recall)

# cm = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
# print(cm)
# cmp = ConfusionMatrixDisplay(
#     confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1)),
#     # display_labels=["class_1", "class_2", "class_3", "class_4"],
# )
#
# cmp.plot(ax=ax)
# plt.show()


# def create_model(c, parameters, kf, Xtrain, Xtest, Ytrain, Ytest,  algorithm, dataset, rs, model, ws):
#     # grid = GridSearchCV(estimator=c, param_grid=parameters, cv=kf, verbose=0, scoring='roc_auc')
#     # grid.fit(Xtrain, Ytrain)
#     # best_model = grid.best_estimator_
#     # print(grid.best_params_)
#
#     etc = ExtraTreesClassifier()
#     _ = etc.fit(X_train, y_train)
#     y_pred = etc.predict(X_test)
#
#     best_model = c
#     best_model.fit(Xtrain, Ytrain)
#     # score = roc_auc_score(Ytest, best_model.predict(Xtest))
#     # print('ROC AUC score:', score)
#     print("\nTEST SET:")
#     # get_scores(Ytest, best_model.predict(Xtest), dataset, algorithm, rs, model, ws, grid.best_params_)
#     multilabel_results(Ytest, best_model.predict(Xtest), algorithm, dataset, rs, model, ws)


def get_scores(y_test, y_pred, algorithm, dataset, rs, model, ws, params=[]):
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
    precision = tp / (tp + fp)
    scores.append(precision)
    print("Precision: " + str(scores[-1]))

    # Sensitivity / recall
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
    # plot_confusion_matrix(cnf_matrix, dataset)
    cmp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred),
        display_labels=["no change", "pic"],
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp.plot(ax=ax)
    fig_name = dataset + '_' + algorithm + '_' + rs + '_' + model + '_' + str(ws) + '.pdf'
    fig.savefig(fig_name)

    # ROC_AUC
    scores.append(roc_auc_score(y_test, y_pred))
    print("ROC AUC score: " + str(scores[-1]))

    scores.append([tn, fp, fn, tp])
    head = ['Dataset', 'Algoritm', 'window', 'model', 'resample', 'F1-Score(micro)', 'F1-Score(macro)',
            'F1-Score(weighted)', 'F1-Score(None)', 'Accuracy', 'precision', 'Sensitivity', 'Specificity',
            'ROC AUC score',
            'Confusion matrix']

    Path("results/cpmp/").mkdir(parents=True, exist_ok=True)
    if not os.path.exists('results/cpmp/' + dataset + '-pic.csv'):
        f = open("results/cpmp/" + dataset + "-pic.csv", "a+")
        writer = csv.writer(f)
        writer.writerow(head)
        f.close()

    f = open("results/cpmp/" + dataset + "-pic.csv", "a+")
    writer = csv.writer(f)
    writer.writerow(scores)
    f.close()

    params_aux = []
    params_aux.append(dataset)
    params_aux.append(algorithm)
    params_aux.append(ws)
    params_aux.append(model)
    params_aux.append(rs)
    params_aux.append(params)
    try:
        f = open("results/cpmp/" + dataset + "-pic-params.csv", "a+")
        writer = csv.writer(f)
        writer.writerow(params_aux)
        f.close()
    except Exception as e:
        print('ERROR saving params in pic: ' + str(e))


    return scores


def create_model(c, parameters, kf, Xtrain, Xtest, Ytrain, Ytest, algorithm, dataset, rs, model, ws):
    grid = GridSearchCV(estimator=c, param_grid=parameters, cv=kf, verbose=0, scoring='roc_auc')
    grid.fit(Xtrain, Ytrain)
    best_model = grid.best_estimator_
    print(grid.best_params_)
    best_model.fit(Xtrain, Ytrain)
    score = roc_auc_score(Ytest, best_model.predict(Xtest))
    print('ROC AUC score:', score)
    print("\nTEST SET:")
    get_scores(Ytest, best_model.predict(Xtest), algorithm, dataset, rs, model, ws, grid.best_params_)


def LogisticRegr_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("\nLOGISTIC REGRESSION")
    parameters = {}
    c = LogisticRegression(random_state=42, n_jobs=-1)
    create_model(c, parameters, kf, Xtrain, Xtest, Ytrain, Ytest, 'LogisticRegression', dataset, rs, model, ws)


def RandomForest_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("RANDOM FOREST")
    parameters = {
        'n_estimators': [int(x) for x in np.linspace(start=100, stop=2000, num=200)],
        'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
    }
    c = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    create_model(c, parameters, kf, Xtrain, Xtest, Ytrain, Ytest, 'RandomForest', dataset, rs, model, ws)


def NN_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("NEURAL NETWORK")
    print("TRAIN AND VALIDATION SETS:")
    parameters = {
        'hidden_layer_sizes': [(1,), (2,), (5,), (10,), (50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        # 'solver': ['adam', 'sgd'],
        # 'learning_rate': ['constant', 'adaptive']
    }

    c = MLPClassifier(random_state=42, max_iter=500)
    create_model(c, parameters, kf, Xtrain, Xtest, Ytrain, Ytest, 'MLP', dataset, rs, model, ws)


def DecisionTree_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print("\nDECISION TREE")
    print("TRAIN AND VALIDATION SETS:")

    parameters = {'max_depth': range(1, 11)}
    c = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    create_model(c, parameters, kf, Xtrain, Xtest, Ytrain, Ytest, 'DT', dataset, rs, model, ws)


def AdaBoost_(Xtrain, Ytrain, Xtest, Ytest, dataset, rs, model, ws):
    print('\nADABOOST')
    print("TRAIN AND VALIDATION SETS:")

    # parameters = {'n_estimators': [10, 50, 100, 500, 1000, 5000], 'learning_rate': arange(0.1, 2.1, 0.1)}
    parameters = {}
    svc = SVC(probability=True, kernel='linear')
    c = AdaBoostClassifier(base_estimator=svc)
    create_model(c, parameters, kf, Xtrain, Xtest, Ytrain, Ytest, 'AB', dataset, rs, model, ws)


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
    # datasets = ['all']
    datasets = ['commons-csv']
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
        "TOTAL_DECLARATIONPARTCHANGES", "TOTAL_CHANGES", "will_change",
        "perf_change"
    ]

    # resamples = ['NONE', 'RUS', 'ENN', 'TL', 'ROS', 'SMOTE', 'ADA']
    # resamples= ['RUS','ENN','TL','ROS','SMOTE','ADA']
    # resamples = ['NONE', 'ROS', 'SMOTE', 'ADA']
    # resamples = ['RUS', 'ENN', 'TL']
    resamples = ['NONE']
    windowsize = [0]
    models = [{'key': 'model1', 'value': model1}, {'key': 'model2', 'value': model2},
              {'key': 'model3', 'value': model3}]
    # models = [{'key':'model4', 'value': model4}, {'key': 'model6', 'value': model6}, {'key': 'model7', 'value': model7}]
    for dataset in datasets:
        for ws in windowsize:
            for rs in resamples:
                for model in models:

                    # to run all datasets
                    # df1 = pd.read_csv(
                    #     '../6.join_metrics/results/commons-csv-perf-diff-all.csv')
                    # df2= pd.read_csv(
                    #     '../6.join_metrics/results/easymock-perf-diff-all.csv')
                    # df_row = pd.concat([df1, df2])

                    # all_releases_df = df_row.fillna(0)

                    all_releases_df = pd.read_csv(
                        '../6.join_metrics/results/' + dataset + '-perf-diff-all-pic.csv')
                    all_releases_df = all_releases_df.fillna(0)
                    # all_releases_df.columns = main_columns
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
                    print("Filtering required columns into X features...")
                    X = all_releases_df[model.get('value')].copy()
                    print("... DONE!")

                    print("Setting y column containing label of change-proneness...")
                    # y = pd.DataFrame(all_releases_df.loc[:, 'perf_change'])
                    # y = pd.DataFrame(all_releases_df.iloc[:, -2:])
                    print(all_releases_df['will_change'].value_counts())
                    print(all_releases_df['perf_change'].value_counts())
                    y = pd.DataFrame(all_releases_df['will_change'] & all_releases_df['perf_change'])
                    print(y.value_counts())
                    print("... DONE!")
                    print("Declaring a dictionary to save results...")
                    results_dict = dict()
                    print("... DONE!")

                    print("Splitting dataset into train and test sets...")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.30, random_state=42)
                    print("General information:")
                    print("X Train set:",
                          X_train.shape[0], "X Test set:", X_test.shape[0])
                    print("y Train set:",
                          y_train.shape[0], "y Test set:", y_test.shape)
                    print("... DONE!")

                    print("Scaling features...")
                    scaler = MinMaxScaler()
                    # X_train = pd.DataFrame(scaler.fit_transform(X_train))
                    # X_test = pd.DataFrame(scaler.fit_transform(X_test))
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.fit_transform(X_test)

                    print("... DONE!")

                    print("Setting stratified k-fold...")
                    k = 10
                    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                    print("k =", k)
                    print("... DONE!\n")
                    # y_test = pd.DataFrame(y_test)
                    # y_train = pd.DataFrame(y_train)
                    y_train = y_train.values.ravel()
                    # WITHOUT OVER OR UNDERSUMPLING
                    # if rs == 'NONE':
                    # RandomForest_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # DecisionTree_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # LogisticRegr_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # NN_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # AdaBoost_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # UNERSAMPLING RUS','ENN','TL'
                    if rs == 'RUS':
                        # X_RUS, y_RUS = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train.values.ravel())
                        X_train, y_train = RandomUnderSampler(random_state=42).fit_resample(X_train,
                                                                                            y_train)
                        # y_RUS = pd.DataFrame(y_RUS)

                        # RandomForest_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # DecisionTree_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # LogisticRegr_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # NN_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # AdaBoost_(X_RUS, y_RUS, X_test, y_test, dataset, rs, model.get('key'), ws)
                    elif rs == 'ENN':
                        # X_ENN, y_ENN = EditedNearestNeighbours(random_state=42).fit_resample(X_train,
                        #                                                                      y_train.values.ravel())
                        X_train, y_train = EditedNearestNeighbours().fit_resample(X_train,
                                                                                  y_train)

                        # RandomForest_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # DecisionTree_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # LogisticRegr_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # NN_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # AdaBoost_(X_ENN, y_ENN, X_test, y_test, dataset, rs, model.get('key'), ws)
                    elif rs == 'TL':
                        # X_TL, y_TL = TomekLinks(random_state=42).fit_resample(X_train, y_train.values.ravel())
                        X_train, y_train = TomekLinks().fit_resample(X_train, y_train)

                        # RandomForest_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # DecisionTree_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # LogisticRegr_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # NN_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # AdaBoost_(X_TL, y_TL, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # OVERSAMPLING 'ROS','SMOTE','ADA'
                    elif rs == 'ROS':
                        ros = RandomOverSampler(random_state=42)
                        # X_ROS, y_ROS = ros.fit_resample(X_train, y_train)
                        X_train, y_train = ros.fit_resample(X_train, y_train)

                        # DecisionTree_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # RandomForest_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # LogisticRegr_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # NN_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # AdaBoost_(X_ROS, y_ROS, X_test, y_test, dataset, rs, model.get('key'), ws)
                    elif rs == 'SMOTE':
                        sm = SMOTE(random_state=42)
                        # X_SMO, y_SMO = sm.fit_resample(X_train, y_train)
                        X_train, y_train = sm.fit_resample(X_train, y_train)
                        # RandomForest_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # DecisionTree_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # LogisticRegr_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # NN_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # AdaBoost_(X_SMO, y_SMO, X_test, y_test, dataset, rs, model.get('key'), ws)
                    elif rs == 'ADA':
                        ada = ADASYN(random_state=42)
                        # X_ADA, y_ADA = ada.fit_resample(X_train, y_train)
                        X_train, y_train = ada.fit_resample(X_train, y_train)
                        # RandomForest_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # DecisionTree_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # LogisticRegr_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # NN_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)
                        # AdaBoost_(X_ADA, y_ADA, X_test, y_test, dataset, rs, model.get('key'), ws)

                    # train models
                    RandomForest_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    DecisionTree_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    LogisticRegr_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    NN_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
                    # AdaBoost_(X_train, y_train, X_test, y_test, dataset, rs, model.get('key'), ws)
