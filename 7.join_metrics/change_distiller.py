import re

import pandas as pd

packs = {'commons-bcel': ['examples\/', 'src\/examples\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/'],
         'commons-io': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/'],
         'junit4': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/', ''],
         'pdfbox': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/',
                    'ant\/src\/main\/java\/', 'examples\/src\/main\/java\/',
                    'lucene\/src\/main\/java\/', 'preflight\/src\/main\/java\/',
                    'pdfbox\/src\/main\/java\/', 'jempbox\/src\/main\/java\/',
                    'xmpbox\/src\/main\/java\/', 'fontbox\/src\/main\/java\/',
                    'fontbox\/src\/test\/java\/', 'debugger\/src\/main\/java\/',
                    'tools\/src\/main\/java\/', 'pdfbox\/src\/test\/java\/',
                    'xmpbox\/src\/test\/java\/', 'examples\/src\/test\/java\/',
                    'preflight\/src\/test\/java\/',
                    ''],
         'wro4j': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/',
                   'wro4j-examples/src/main/java/', 'wro4j-maven-plugin/src/main/java/',
                   'wro4j-extensions/src/main/java/', 'wro4j-test-utils/src/main/java/',
                   'wro4j-core/src/main/java/',
                   'wro4j-examples/src/test/java/', 'wro4j-maven-plugin/src/test/java/',
                   'wro4j-extensions/src/test/java/', 'wro4j-test-utils/src/test/java/',
                   'wro4j-core/src/test/java/', '']
        }

def format_method(row):
    method_name = row['CLASS_PREVIOUSNAME']

    proj = row['PROJECT_NAME']
    for p in packs[proj]:
        pattern = r'\/project[AB]\/' + p
        name = re.split(pattern, method_name)
        if len(name) > 1:
            method_name = name[1]

    method_name = method_name.replace('.java', '')
    method_name = method_name.replace('/', '.')

    return method_name


def join_change_distiller(project_name, current_hash):
    # print("Change distiller")
    csv_path = '../6.cd/results/' + project_name + '-cd-methods-results.csv'

    metrics = ["PROJECT_NAME", "PREVIOUS_COMMIT", "CURRENT_COMMIT", "CLASS_CURRENTCOMMIT", "CLASS_PREVIOUSCOMMIT",
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
               "TOTAL_DECLARATIONPARTCHANGES", "TOTAL_CHANGES"]

    df = pd.read_csv(csv_path,  sep=',', engine='python', index_col=False)
    df.columns = metrics
    df = df[(df['CURRENT_COMMIT'] == current_hash)]

    if len(df.index):
        df['method_name'] = df.apply(format_method, axis=1)

    # print(df.shape[0])
    return df
