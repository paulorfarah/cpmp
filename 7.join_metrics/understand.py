import pandas as pd

def join_understand(project_name, current_hash):
    # print("Understand ")
    csv_path = '../2.understand/results/' + project_name + '/' + current_hash + '.csv'

    # understandRepo = pydriller.Git(csvPathUndestand)
    metrics = ["Kind", "Name", "File", "AvgCyclomatic", "AvgCyclomaticModified", "AvgCyclomaticStrict",
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
                         "SumCyclomatic", "SumCyclomaticModified", "SumCyclomaticStrict", "SumEssential"]

    df = pd.read_csv(csv_path, usecols=metrics, sep=',', engine='python', index_col=False)
    df_methods = df[df['Kind'].str.contains("Method")]
    df_constructors = df[df['Kind'].str.contains("Constructor")]

    df_filtered = pd.concat([df_methods, df_constructors])

    # print(df_filtered.shape[0])
    return df_filtered
