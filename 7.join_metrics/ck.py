import pandas as pd


def join_ck(project_name, current_hash):
    print("CK ")
    csv_path = '../1.ck/results/' + project_name + '-methods-results.csv'

    method_metrics = ['file', 'class', 'method', 'constructor', 'line', 'cbo', 'cboModified', 'fanin', 'fanout', 'wmc',
                      'rfc', 'loc', 'returnsQty', 'variablesQty', 'parametersQty', 'methodsInvokedQty',
                      'methodsInvokedLocalQty', 'methodsInvokedIndirectLocalQty', 'loopQty', 'comparisonsQty',
                      'tryCatchQty', 'parenthesizedExpsQty', 'stringLiteralsQty', 'numbersQty', 'assignmentsQty',
                      'mathOperationsQty', 'maxNestedBlocksQty', 'anonymousClassesQty', 'innerClassesQty', 'lambdasQty',
                      'uniqueWordsQty', 'modifiers', 'logStatementsQty', 'hasJavaDoc']



    # metrics = ["class", "type", "cbo", "wmc", "dit", "rfc", "lcom", "tcc", "lcc", "totalMethodsQty",
    #                      "staticMethodsQty", "publicMethodsQty", "privateMethodsQty", "protectedMethodsQty",
    #                      "defaultMethodsQty", "abstractMethodsQty", "finalMethodsQty", "synchronizedMethodsQty",
    #                      "totalFieldsQty", "staticFieldsQty", "publicFieldsQty", "privateFieldsQty",
    #                      "protectedFieldsQty", "defaultFieldsQty", "visibleFieldsQty", "finalFieldsQty",
    #                      "synchronizedFieldsQty", "nosi", "loc", "returnQty", "loopQty", "comparisonsQty",
    #                      "tryCatchQty", "parenthesizedExpsQty", "stringLiteralsQty", "numbersQty", "assignmentsQty",
    #                      "mathOperationsQty", "variablesQty", "maxNestedBlocksQty", "anonymousClassesQty",
    #                      "innerClassesQty", "lambdasQty", "uniqueWordsQty", "modifiers", "logStatementsQty"]


    df = pd.read_csv(csv_path, usecols=method_metrics, sep=',', index_col=False)

    print(df.shape[0])
    return df

