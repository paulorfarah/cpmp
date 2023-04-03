import re

import pandas as pd

packs = {'commons-bcel': 'org',
         'commons-io': 'org',
         'junit': 'org',
         'pdfbox': 'org',
         'wro4j': 'ro.isdc.wro',
         }


def format_method(row):
    method_name = row['method'].replace('/', '.')

    proj = row['project']
    pack_name = packs[proj]
    method_name = re.findall(pack_name + '.+', method_name)[0]

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


    df['method_name'] = df.apply(format_method, axis=1)
    # print(df.shape[0])
    return df
