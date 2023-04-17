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
                      'src\/test\/java\/', 'src\/java\/', 'src\/main\/java\/', 'src\/test\/',
                      'easymock\/src\/samples\/java\/',
                      'easymock\/bench\/src\/main\/java\/', 'easymock\/core\/src\/main\/java\/', 'easymock\/core\/src\/samples\/java\/',
                      'easymock\/core\/src\/test\/java\/', 'easymock\/test-android\/src\/main\/java\/', 'easymock\/test-deploy\/src\/main\/java\/',
                      'easymock\/test-integration\/src\/main\/java\/', 'easymock\/test-integration\/src\/test\/java\/',
                      'easymock\/test-java8\/src\/test\/java\/', 'easymock\/test-junit5\/src\/test\/java\/', 'easymock\/test-nodeps\/src\/test\/java\/',
                      'easymock\/test-osgi\/src\/test\/java\/', 'easymock\/test-testng\/src\/test\/java\/',
                      'easymock\/src\/test\/java\/', 'easymock\/src\/java\/', 'easymock\/src\/main\/java\/', 'easymock\/src\/test\/',
                      '']
    }

def format_method(row):
    # public java.lang.String org.jivesoftware.openfire.IQHandlerInfo.getNamespace()

    method_name = row['method_name'].split(' ')[-1]

    patternMethod = r'[a-zA-Z._]+\([a-zA-Z ._]+\)'
    method = re.split(patternMethod, method_name)
    if len(method):
        method_name = method[0]

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

project_name = 'Openfire'
# df = pd.read_csv('results/' + project_name + '-all-releases.csv')
df = pd.read_csv('results/commons-bcel-all-releases.csv')

df_perf = pd.read_csv('../3.performance/results/' + project_name + '/' + project_name + '-method-performance-median_filtered.csv')

df_perf.columns = ['commit_hash', 'prevcommit', 'class_name', 'method_name', 'metric', 'stat', 'pvalue', 'mean_val',
                   'avg2', 'change', 'perf_change', 'change_abs', 'changed_median']
df_perf['project'] = project_name
if len(df_perf):
    df_perf['method_name'] = df_perf.apply(format_method, axis=1)
