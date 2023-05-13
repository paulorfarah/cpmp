from mdi import mdi
from information_gain import information_gain_cpmp


if __name__ == "__main__":
    projects = ['commons-bcel', 'commons-csv', 'commons-io', 'easymock', 'Openfire', 'pdfbox', 'wro4j']

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
        'commitprevious', 'release', 'file', 'method', 'BOC', 'TACH', 'FCH', 'LCH', 'CHO',
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
    #main_columns: file, class ,method, constructor, line, cbo, cboModified, fanin, fanout, wmc, rfc, loc, returnsQty, variablesQty, parametersQty, methodsInvokedQty, methodsInvokedLocalQty, methodsInvokedIndirectLocalQty, loopQty, comparisonsQty, tryCatchQty, parenthesizedExpsQty, stringLiteralsQty, numbersQty, assignmentsQty, mathOperationsQty, maxNestedBlocksQty, anonymousClassesQty, innerClassesQty, lambdasQty, uniqueWordsQty, modifiers, logStatementsQty, hasJavaDoc, method_name, current_hash, Kind, Name, File, AvgCyclomatic, AvgCyclomaticModified, AvgCyclomaticStrict, AvgEssential, AvgLine, AvgLineBlank, AvgLineCode, AvgLineComment, CountClassBase, CountClassCoupled, CountClassDerived, CountDeclClass, CountDeclClassMethod, CountDeclClassVariable, CountDeclFile, CountDeclFunction, CountDeclInstanceMethod, CountDeclInstanceVariable, CountDeclMethod, CountDeclMethodAll, CountDeclMethodDefault, CountDeclMethodPrivate, CountDeclMethodProtected, CountDeclMethodPublic, CountInput, CountLine, CountLineBlank, CountLineCode, CountLineCodeDecl, CountLineCodeExe, CountLineComment, CountOutput, CountPath, CountSemicolon, CountStmt, CountStmtDecl, CountStmtExe, Cyclomatic, CyclomaticModified, CyclomaticStrict, Essential, MaxCyclomatic, MaxCyclomaticModified, MaxCyclomaticStrict, MaxEssential, MaxInheritanceTree, MaxNesting, PercentLackOfCohesion, RatioCommentToCode, SumCyclomatic, SumCyclomaticModified, SumCyclomaticStrict, SumEssential, project, commit, commitprevious, release, file, method, BOC, TACH, FCH, LCH, CHO, FRCH, CHD, WCH, WCD, WFR, ATAF, LCA, LCD, CSB, CSBS, ACDF, PROJECT_NAME, PREVIOUS_COMMIT, CURRENT_COMMIT, CLASS_CURRENTCOMMIT, CLASS_PREVIOUSCOMMIT, CLASS_CURRENTNAME, CLASS_PREVIOUSNAME, STATEMENT_DELETE, STATEMENT_INSERT, STATEMENT_ORDERING_CHANGE, STATEMENT_PARENT_CHANGE, STATEMENT_UPDATE, TOTAL_STATEMENTLEVELCHANGES, PARENT_CLASS_CHANGE, PARENT_CLASS_DELETE, PARENT_CLASS_INSERT, CLASS_RENAMING, TOTAL_CLASSDECLARATIONCHANGES, RETURN_TYPE_CHANGE, RETURN_TYPE_DELETE, RETURN_TYPE_INSERT, METHOD_RENAMING, PARAMETER_DELETE, PARAMETER_INSERT, PARAMETER_ORDERING_CHANGE, PARAMETER_RENAMING, PARAMETER_TYPE_CHANGE, TOTAL_METHODDECLARATIONSCHANGES, ATTRIBUTE_RENAMING, ATTRIBUTE_TYPE_CHANGE, TOTAL_ATTRIBUTEDECLARATIONCHANGES, ADDING_ATTRIBUTE_MODIFIABILITY, REMOVING_ATTRIBUTE_MODIFIABILITY, REMOVING_CLASS_DERIVABILITY, REMOVING_METHOD_OVERRIDABILITY, ADDING_CLASS_DERIVABILITY, ADDING_CLASS_DERIVABILITY, ADDING_METHOD_OVERRIDABILITY, TOTAL_DECLARATIONPARTCHANGES, TOTAL_CHANGES, will_change

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
        'BOC', 'TACH', 'FCH', 'LCH', 'CHO', 'FRCH', 'CHD', 'WCH', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS',
        'ACDF',
    ]
    change_distiller_metrics = [
        "TOTAL_CHANGES"
        "will_change"
    ]

    model1 = structural_metrics
    model2 = structural_metrics + evolutionary_metrics
    model3 = evolutionary_metrics

    # for project_name in projects:
    # mdi(projects, main_columns, model3)
    information_gain_cpmp(projects, main_columns, model3)
