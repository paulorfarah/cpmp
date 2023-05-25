import argparse
import collections
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

pd.set_option('display.max_columns', None)

# val_cols = ['commit_hash', 'class_name', 'method_name', 'own_duration', 'cumulative_duration', 'active',
#             'available', 'buffers', 'cached ', 'child_major_faults', 'child_minor_faults', 'commit_limit',
#             'committed_as', 'cpu_percent', 'data', 'dirty', 'free', 'high_free', 'high_total', 'huge_pages_total',
#             'huge_pages_free', 'huge_pages_total1', 'hwm', 'inactive', 'laundry', 'load1', 'load5', 'load15',
#             'locked', 'low_free', 'low_total', 'major_faults', 'mapped', 'mem_percent', 'minor_faults',
#             'page_tables', 'pg_fault', 'pg_in', 'pg_maj_faults', 'pg_out', 'read_bytes', 'read_count', 'rss',
#             'shared', 'sin', 'slab', 'sout', 'sreclaimable', 'stack', 'sunreclaim', 'swap', 'swap_cached',
#             'swap_free', 'swap_total', 'swap_used', 'swap_used_percent ', 'total', 'used', 'used_percent', 'vm_s',
#             'vmalloc_chunk', 'vmalloc_total', 'vmalloc_used', 'wired', 'write_back', 'write_back_tmp',
#             'write_bytes', 'write_count']

val_cols = ['commit_hash', 'class_name', 'own_duration_avg']


# commit_hash,class_name,method_name,own_duration,committer_date

def read_csv(file):
    # cols = ['methods.id', 'committer_date', 'commit_hash', 'run', 'class_name',
    #         'method_name', 'method_started_at', 'method_ended_at',
    #         'caller_id', 'own_duration', 'cumulative_duration', 'timestamp ',
    #         'active', 'available', 'buffers', 'cached ', 'child_major_faults', 'child_minor_faults', 'commit_limit',
    #         'committed_as', 'cpu_percent', 'data', 'dirty', 'free', 'high_free', 'high_total', 'huge_pages_total',
    #         'huge_pages_free',
    #         'huge_pages_total1', 'hwm', 'inactive', 'laundry', 'load1', 'load5', 'load15', 'locked', 'low_free',
    #         'low_total', 'major_faults',
    #         'mapped', 'mem_percent', 'minor_faults', 'page_tables', 'pg_fault', 'pg_in', 'pg_maj_faults', 'pg_out',
    #         'read_bytes', 'read_count',
    #         'rss', 'shared', 'sin', 'slab', 'sout', 'sreclaimable', 'stack', 'sunreclaim', 'swap', 'swap_cached',
    #         'swap_free', 'swap_total', 'swap_used',
    #         'swap_used_percent ', 'total', 'used', 'used_percent', 'vm_s', 'vmalloc_chunk', 'vmalloc_total',
    #         'vmalloc_used', 'wired',
    #         'write_back', 'write_back_tmp', 'write_bytes', 'write_count']
    cols = ['committer_date', 'commit_hash', 'class_name', 'method_name',  'own_duration_avg']
    df = pd.read_csv(file, names=cols, sep=';', header=None)
    return df


def calculate_perf_change(commit_hash, median, medianDurations):
    if median > medianDurations[commit_hash]:
        return 1
    else:
        return 0


def own_duration_avg_by_class(project_name, file, versions):
    df = read_csv(file)
    # df_res = pd.DataFrame(pd.np.empty((0, 8)))
    # df_res.columns = ['commit', 'class_name', 'metric', 'avg', 'sum', 'max', 'min', 'stddev']
    min = df.groupby(['commit_hash', 'class_name'])['own_duration_avg'].min().reset_index(name='min_val')
    max = df.groupby(['commit_hash', 'class_name'])['own_duration_avg'].max().reset_index(name='max_val')
    mean = df.groupby(['commit_hash', 'class_name'])['own_duration_avg'].mean().reset_index(name='mean_val')
    median = df.groupby(['commit_hash', 'class_name'])['own_duration_avg'].median().reset_index(name='median_val')
    stddev = df.groupby(['commit_hash', 'class_name'])['own_duration_avg'].std().reset_index(name='stddev_val')

    # res = pd.concat([min, max, mean, median, stddev])
    res = pd.merge(min, max, on=['commit_hash', 'class_name'])
    res = pd.merge(res, mean, on=['commit_hash', 'class_name'])
    res = pd.merge(res, median, on=['commit_hash', 'class_name'])
    res = pd.merge(res, stddev, on=['commit_hash', 'class_name'])

    # res['perf_changed'] = 0
    # #calculates if median of one instance is greater than median of all
    # df_median_by_commit = df.groupby(['commit_hash'])['own_duration_avg'].median()#.reset_index(name='median_by_commit')
    #
    # medianDurations = {}
    # for commit, commit_median in df_median_by_commit.iteritems():
    #     print(commit + ': ' + str(commit_median))
    #     df_commit = df.loc[df['commit_hash'] == commit]
    #     medianDurations[commit] = df_commit['own_duration_avg'].median()
    #     # res.loc[res['commit_hash'] == commit]['performance_changed'] = np.where(res['median'] > medianDurations[commit], 1, 0)
    # res['perf_changed'] = res.apply(lambda x: calculate_perf_change(x.commit_hash, x.median_val, medianDurations), axis=1)
    # res.loc[res['perf_changed'] == 1].to_csv('results/' + project_name + '/' + project_name + '-class-performance-median_filtered.csv', index=False)

    res.to_csv('results/' + project_name + '/' + project_name + '-class-performance-avg.csv', index=False)


def student_ttest_by_class(project_name, file, versions):
    # https://analyticsindiamag.com/a-beginners-guide-to-students-t-test-in-python-from-scratch%EF%BF%BC/
    # https://stackoverflow.com/questions/13404468/t-test-in-pandas

    df = read_csv(file)
    df_res = pd.DataFrame(pd.np.empty((0, 10)))
    df_res.columns = ['commit_hash', 'prevcommit', 'class_name', 'metric', 'stat', 'pvalue', 'mean_val', 'avg2',
                      'change', 'perf_change']
    # df_res.columns = ['committer_date', 'commit_hash', 'class_name', 'own_duration_avg']

    for v2 in range(1, len(versions)):
        v1 = v2 - 1
        df1 = df.query("commit_hash == '" + versions[v1] + "'")
        # df1 = df.groupby('a')['b'].apply(list).reset_index(name='new')
        grouped1 = df1.groupby(['commit_hash', 'class_name'])['own_duration_avg'].apply(list).reset_index(name='new')

        # for metric in val_cols[2:]:
        metric = 'own_duration_avg'
        print("---------------- " + metric + " -------------------------")
        # rows = grouped1[metric]

        for name, value in grouped1.iterrows():
            # print(len(values1))
            # for i in values1:
            #     print(i, type(i))
            commit = value[0]
            class_name = value[1]
            # vals1 = values1
            vals1 = value[2]
            # try:
            #     # print('1->', values1.iloc[0])
            #     if type(values1.iloc[0]) is str:
            #         vals1 = [eval(i) for i in values1]
            # except:
            #     print(sys.exc_info())
            #     print('values1 empty')
            #     print(values1)
            vals2 = df.loc[(df['commit_hash'] == versions[v2]) &
                           (df['class_name'] == class_name)][metric].to_list()  # &
            # (df['method_name'] == name[2])][metric]

            # if vals2.any():
            # print1
            # vals2 = values2
            # try:
            #     # print('2->', values2.iloc[0])
            #     if type(values2.iloc[0]) is str:
            #         vals2 = [eval(i) for i in values2]
            # except:
            #     print(sys.exc_info())
            #     print('values2 empty')
            #     print(values1)
            try:
                if isinstance(vals1, collections.abc.Sequence) and isinstance(vals2, collections.abc.Sequence):
                    stat, pvalue = ttest_ind(vals1, vals2)
                    # print(commit + ' ' + class_name + ' ' + str(pvalue))
                else:
                    stat = -1
                    pvalue = -1
            except ZeroDivisionError:
                print('ZeroDivisionError1: ', vals1, vals2)
                stat = 0
                pvalue = 100

            perf_change = 0
            if pvalue <= 0.05:
                perf_change = 1
            try:
                avg1 = sum(vals1) / len(vals1)
            except ZeroDivisionError:
                print('ZeroDivisionError2: ', vals1)
                avg1 = 0
            except:
                print('error1: ', sys.exc_info())
                print('vals1: ', len(vals1))
            try:
                avg2 = sum(vals2) / len(vals2)
            except ZeroDivisionError:
                print('ZeroDivisionError3: ', vals2)
                avg2 = 0
            except:
                print('error2: ', sys.exc_info())
                print('vals2: ', vals2)

            try:
                change = round(((avg2 - avg1) / avg1 * 100), 2)
                # change = 100 * abs((avg1 - avg2) / ((avg1+avg2)/2))
            except ZeroDivisionError:
                print('ZeroDivisionError4: ', avg1)
                change = 0
            df_res.loc[len(df_res.index)] = [versions[v1], versions[v2], class_name, metric, stat,
                                             pvalue, avg1, avg2, change, perf_change]
            # print([versions[v1], versions[v2], class_name, metric, stat,
            #                                  pvalue, avg1, avg2, change, perf_change])

    # delete avg with zero value
    df_res.drop(df_res.index[df_res['mean_val'] == 0], inplace=True)
    df_res.drop(df_res.index[df_res['avg2'] == 0], inplace=True)
    df_res['change_abs'] = df_res['change'].abs()
    df_median_by_commit = df_res.groupby(['commit_hash'])['change_abs'].median()
    print(df_median_by_commit.head)
    medianDurations = {}
    for commit, commit_median in df_median_by_commit.iteritems():
        print(commit + ': ' + str(commit_median))
        df_commit = df_res.loc[df['commit_hash'] == commit]
        medianDurations[commit] = df_commit['change_abs'].median()
    # df_res['changed_median'] = df_res.apply(
    df_res2 = df_res.apply(
        lambda x: calculate_perf_change(x.commit_hash, abs(x.change), medianDurations), axis=1)

    if not df_res2.empty:
        df_res['changed_median'] = df_res2
        df_res.to_csv('results/' + project_name + '/' + project_name + '-class-performance-diff_all.csv', index=False)
        df_res.loc[df_res['perf_change'] == 1].to_csv(
            'results/' + project_name + '/' + project_name + '-class-performance-diff_filtered.csv', index=False)
        df_res.loc[df_res['changed_median'] == 1].to_csv(
            'results/' + project_name + '/' + project_name + '-class-performance-median_filtered.csv', index=False)


def student_ttest_by_method(project_name, file, versions):
    # df = read_csv(file)
    df = pd.read_csv(file)
    # df_res = pd.DataFrame(pd.np.empty((0, 9)))
    df_res = pd.DataFrame(columns=['commit', 'prevcommit', 'class_name', 'method_name', 'metric', 'stat', 'pvalue',
                                   'avg1', 'avg2', 'change'])
    # df_res.columns = ['commit', 'prevcommit', 'class_name', 'method_name', 'metric', 'stat', 'pvalue', 'avg1', 'avg2',
                      #'change']
    # df_res.columns = ['committer_date', 'commit_hash', 'class_name', 'own_duration_avg']

    for v2 in range(1, len(versions)):
        v1 = v2 - 1
        df1 = df.query("commit_hash == '" + versions[v1] + "'")
        # df1 = df.groupby('a')['b'].apply(list).reset_index(name='new')
        grouped1 = df1.groupby(['commit_hash', 'class_name', 'method_name'])['own_duration'].apply(
            list).reset_index(name='new')

        # for metric in val_cols[2:]:
        metric = 'own_duration'
        print("---------------- " + metric + " -------------------------")
        # rows = grouped1[metric]

        for name, value in grouped1.iterrows():
            # print(len(values1))
            # for i in values1:
            #     print(i, type(i))
            commit = value[0]
            class_name = value[1]
            method_name = value[2]
            vals1 = value[3]

            vals2 = df.loc[(df['commit_hash'] == versions[v2]) &
                           (df['class_name'] == class_name) &
                           (df['method_name'] == value[2])][metric].to_list()
            try:
                if isinstance(vals1, collections.abc.Sequence) and isinstance(vals2, collections.abc.Sequence):
                    stat, pvalue = ttest_ind(vals1, vals2)
                    # print(commit + ' ' + class_name + ' ' + str(pvalue))
                else:
                    stat = -1
                    pvalue = -1
            except ZeroDivisionError:
                print('ZeroDivisionError1: ', vals1, vals2)
                stat = 0
                pvalue = 100
            if pvalue <= 0.05:
                try:
                    avg1 = sum(vals1) / len(vals1)
                except ZeroDivisionError:
                    print('ZeroDivisionError2: ', vals1)
                    avg1 = 0
                except:
                    print('error1: ', sys.exc_info())
                    print('vals1: ', len(vals1))
                try:
                    avg2 = sum(vals2) / len(vals2)
                except ZeroDivisionError:
                    print('ZeroDivisionError3: ', vals2)
                    avg2 = 0
                except:
                    print('error2: ', sys.exc_info())
                    print('vals2: ', vals2)

                try:
                    change = round(((abs(avg2 - avg1) / avg1) * 100), 2)
                except ZeroDivisionError:
                    print('ZeroDivisionError4: ', avg1)
                    change = 100
                df_res.loc[len(df_res.index)] = [versions[v1], versions[v2], class_name, method_name, metric, stat,
                                                 pvalue, avg1, avg2, change]
                print([versions[v1], versions[v2], class_name, method_name, metric, stat,
                       pvalue, avg1, avg2, change])
    df_res.to_csv('results/method_' + project_name + '-performance-diff.csv', index=False)


def main():
    print('starting...')
    commits = {'commons-bcel': ['a9c13ede0e565fae0593c1fde3b774d93abf3f71', 'bebe70de81f2f8912857ddb33e82d3ccc146a24e',
                                'bbaf623d750f030d186ed026dc201995145c63ec', 'fa271c5d7ed94dd1d9ef31c52f32d1746d5636dc',
                                'dce57b377d1ad6711ff613639303683e90f7bcc8', '9174edf0d530540c9f6df76b4d786c5a6ad78a5d',
                                '3aecd517ad0ac4c83828a5f89b6b062acb6f4f6a', 'f38847e90714fbefc33042912d1282cc4fb7d43e'],
               'commons-csv': ['73a432514936fcee1386b37a0d60dcd913706bd1', 'e7142a3937825074ec68e4bacba30f9c962bd1e4',
                               'df479df17676148bf6391401a704a7d7265c45fa', '399d2929ee0912bfeda8a4ef125f1b96bb7fd144',
                               'f3b8653021830d8a503c5e7a9d33cb82b16db739', 'f2c58503d76be1dfb8072f6a7d592f88133708e1',
                               '41b194c718d50763a79951029787cca70a5804a5', '1447159b926076c9222a96b3abbe17571953a74f',
                               '4f0daa3bb2a5fa28286f1973deb9d13996cc73cc', 'bf32c9102fb1b5fdfa7a26a120b5d9a6b428dd2f'],
               'commons-text': ['cb85bed468e99d34b88d0c81fe20eb3b1615660e',
                                'b63df8d66e8306b2608c16be3661248348e78a2f',
                                '3866d2626dc3767b003c7cfe163a388b10c80957',
                                '7643b12421100d29fd2b78053e77bcb04a251b2e',
                                '4736b16d0e644289f3106275ebb1315750234e40',
                                'e1af89b53855f2f19138cbc3e8a49ca179c3d8f8',
                                'd3d93c4e68ce5d8c25aecbfff9d17017594bf3f2',
                                'ba44287bdd17a709523364820495760645da85b9'],
               'easymock': ['1a01c13b73c0c66de1efa3db4d73a839aaf20ab9',
                            '266c64660523d728592e646fa9f3f3e2fdfdbc4a',
                            'caf80a128f00481e8c19151257001015acc3e76e',
                            'a6e7c7e6fc54c8ee3dce10edbe76c1821f10cd92',
                            '0c45595df8f8a0939dbc0b0385c8afe7502b1190',
                            '853c1e35326a54e3fc28177c5c84c07652750140',
                            '3506ccdfa91500016e3a0908d7ccabc171aa5602',
                            '22ade6817ad07f22f1d8f0263ff6ddc6fc9b05db',
                            '36782213bf5e8f1e0f601cb73774ec7a5a8c58f1'],
               'jgit': ['73a432514936fcee1386b37a0d60dcd913706bd1',
                        'e7142a3937825074ec68e4bacba30f9c962bd1e4',
                        'df479df17676148bf6391401a704a7d7265c45fa',
                        '399d2929ee0912bfeda8a4ef125f1b96bb7fd144',
                        'f3b8653021830d8a503c5e7a9d33cb82b16db739',
                        'f2c58503d76be1dfb8072f6a7d592f88133708e1',
                        '41b194c718d50763a79951029787cca70a5804a5',
                        '1447159b926076c9222a96b3abbe17571953a74f',
                        '4f0daa3bb2a5fa28286f1973deb9d13996cc73cc',
                        'bf32c9102fb1b5fdfa7a26a120b5d9a6b428dd2f',
                        '29159f1171c4930128ab0557836e856ce8cba6c8'],
               'Openfire': ['c0f2614b1bcfbdcad337128978aa9ae82e66fc50',
                            'd7dfd04eca85ca8933a408c7bc99eb6ed8fb8599',
                            'e020f58943742b0be541bafd30808d587867712b',
                            'f5aece5bbdd81bbab824e3995cbdfd96920766f2',
                            '5b6b732fda44fbc3c38a7036b1ed7522d9f8b129',
                            '348185c8385cf8923455a71dd614d7ee09ee7bf0',
                            '1bcb80f9f23d70fbd7ee8a833abe02b2305f848e',
                            'cb90fc79edb71b888d3ee25f2f59da3be2e72174',
                            'c4c568e13715a3cdb980e340352ec4fb30c61fae',
                            '2bc37e6f1d48c84228efe2ccd360148cc9be46f0',
                            '9b35e20028cdb52e29e4ac815a8419ffb35ca2d3',
                            'b61bce39e3a0ce786e4464706ea2e9b4c5be7a77',
                            '18a7bb6262b4237f9c80071d24234f5d9b2e08fc',
                            '13e73ed3d7fc00646c642d02b66b50a86cd1c42b',
                            '18257f671cef3c8340be950a4d5c50ef3b31d9f6',
                            '92a9f2e9e2c78aabdf446af0c680c7c0991f807a',
                            '41041642078a1e498c4f8b6e6a5b226409b5dbcd']
               }

    # projects = ['commons-bcel', 'commons-csv', 'commons-text', 'easymock', 'jgit', 'Openfire']
    projects = ['commons-bcel']
    for project_name in projects:
        file = 'data/' + project_name + '/own_dur_trace-all.csv'
        commits_list = commits[project_name]

        # own_duration_avg_by_class(project_name, file, commits_list)
        # student_ttest_by_class(project_name, file, commits_list)
        student_ttest_by_method(project_name, file, commits_list)


if __name__ == "__main__":
    main()
    print('end.')
