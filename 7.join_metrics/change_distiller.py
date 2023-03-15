import pandas as pd


def join_change_distiller(project_name, current_hash):
    print("Change distiller")
    csv_path = '../6.cd/results/' + project_name + '-methods-results.csv'

    metrics = ["PROJECT_NAME", "CURRENT_COMMIT", "PREVIOUS_COMMIT", "CLASS_CURRENTCOMMIT",
                             "CLASS_PREVIOUSCOMMIT",
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
                             "TOTAL_DECLARATIONPARTCHANGES", "TOTAL_CHANGES"]


    df = pd.read_csv(csv_path, usecols=metrics, sep=',', engine='python', index_col=False)
    df = df[(df['CURRENT_COMMIT'] == current_hash)]

    print(df.shape[0])
    return df