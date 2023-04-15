import re
import pandas as pd

packs = {'commons-bcel': ['examples\/', 'src\/examples\/', 'src\/java\/', 'src\/main\/java\/',
                                              'src\/test\/', 'src\/test\/java\/'],
         'commons-io': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/'],
         'junit4': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/', ''],
         'pdfbox': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/',
                    'ant\/src\/main\/java\/', 'examples\/src\/main\/java\/',
                    'lucene\/src\/main\/java\/', 'preflight\/src\/main\/java\/',
                    'pdfbox\/src\/main\/java\/', 'jempbox\/src\/main\/java\/',
                    'xmpbox\/src\/main\/java\/', 'fontbox\/src\/main\/java\/',
                    'fontbox\/src\/test\/java\/', 'debugger\/src\/main\/java\/',
                    'tools\/src\/main\/java\/', 'pdfbox\/src\/test\/java\/',
                    'xmpbox\/src\/test\/java\/', 'examples\/src\/test\/java',
                    'preflight\/src\/test\/java\/',
                    ''],
         'wro4j': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/',
                'wro4j-examples\/src\/main\/java\/', 'wro4j-maven-plugin\/src\/main\/java\/',
                   'wro4j-extensions\/src\/main\/java\/', 'wro4j-test-utils\/src\/main\/java\/',
                   'wro4j-core\/src\/main\/java\/',
                   'wro4j-examples\/src\/test\/java\/', 'wro4j-maven-plugin\/src\/test\/java\/',
                   'wro4j-extensions\/src\/test\/java\/', 'wro4j-test-utils\/src\/test\/java\/',
                   'wro4j-core\/src\/test\/java\/', ''],
         'commons-csv': ['src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/', ''],
         'easymock': ['src\/samples\/java\/',
                      'bench\/src\/main\/java\/', 'core\/src\/main\/java\/', 'core\/src\/samples\/java\/',
                      'core\/src\/test\/java\/', 'test-android\/src\/main\/java\/', 'test-deploy\/src\/main\/java\/',
                      'test-integration\/src\/main\/java\/', 'test-integration\/src\/test\/java\/',
                      'test-java8\/src\/test\/java\/', 'test-junit5\/src\/test\/java\/', 'test-nodeps\/src\/test\/java\/',
                      'test-osgi\/src\/test\/java\/', 'test-testng\/src\/test\/java\/',
                      'src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/', '']
    }


def format_method(row):
    method_name = row['method']

    proj = row['project']
    for p in packs[proj]:
        pattern = r'^\/' + p
        name = re.split(pattern, method_name)
        if len(name) > 1:
            method_name = name[1]
            break

    method_name = method_name.replace('/', '.')
    method_name = method_name.replace(' ', '')
    return method_name


def join_evo(project_name, current_hash):
    # print("evolutive")
    csv_path = '../5.evometrics/results/' + project_name + '-methods-results-processMetrics.csv'

    metrics = ['project', 'commit', 'commitprevious', 'release', 'file', 'method', 'BOC', 'TACH', 'FCH', 'LCH', 'CHO',
               'FRCH', 'CHD', 'WCH', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS', 'ACDF']

    df = pd.read_csv(csv_path, usecols=metrics, sep=',', engine='python', index_col=False)
    df = df[(df['commit'] == current_hash)]

    # '/src/test/java/org/apache/commons/io/DirectoryWalkerTestCaseJava4/testFilterAndLimitC()'
    # evo: '/src/test/org/apache/commons/io/FileUtilsTestCase/testForceDeleteDir()'
    # ck:  'org.apache.commons.io.FileUtilsTestCase.testForceDeleteDir()'

    if len(df):
        df['method_name'] = df.apply(format_method, axis=1)
    return df
