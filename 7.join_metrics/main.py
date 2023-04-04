
import pydriller
import argparse
import git
import pandas as pd
import numpy as np

from ck import join_ck
from understand import join_understand
from evo import join_evo
from change_distiller import join_change_distiller
from smells import join_smells

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Join Metrics')
    # args = ap.parse_args()

    projects = ['commons-bcel']
    for project_name in projects:
        repo_path = "repos/" + project_name
        gr = pydriller.Git(repo_path)
        repo = git.Repo(repo_path)
        tags = repo.tags
        release = 1

        csv_results = 'results/' + project_name + '-all-releases.csv'

        # f = open(csvPath, "w")
        # writer = csv.writer(f)
        missing = []
        for tag in tags:
            current_hash = gr.get_commit_from_tag(tag.name).hash

            try:
                df_joined = join_ck(project_name, current_hash)
                df_joined['commit'] = current_hash

                und = join_understand(project_name, current_hash)
                df_joined = pd.merge(left=df_joined, right=und, on='method_name', how='outer', indicator=True)
                df_disjoint = df_joined.query('_merge != "both"')[['method_name', 'Name']]

                # df_joined = pd.merge(left=df_joined, right=und, on='method_name', how='inner')

                evo = join_evo(project_name, current_hash)
                df_joined = pd.merge(left=df_joined, right=evo, on='method_name', how='inner')
                # df_disjoint = df_joined.query('_merge != "both"')[['method_name']]

                # smells = join_smells(project_name, current_hash)
                # df_joined = pd.merge(left=df_joined, right=smells, left_on='class', right_on='fullyQualifiedName')

                # merged_full = pd.merge(left=ck_understand_process_organic, right=releaseChangeDistillerMetrics,
                #                        left_on='class', right_on='CLASS_PREVIOUSCOMMIT')

                change_distiller = join_change_distiller(project_name, current_hash)
                df_joined = pd.merge(left=df_joined, right=change_distiller,
                                     left_on='class', right_on='CLASS_PREVIOUSCOMMIT')

                # df_joined.loc[:,'class_frequency'] = 1
                df_joined.loc[:, 'will_change'] = 0
                # df_joined.loc[:,'number_of_changes'] = 0
                df_joined.loc[:, 'release'] = release
                medianChanges = df_joined['TOTAL_CHANGES'].median()
                df_joined['will_change'] = np.where(df_joined['TOTAL_CHANGES'] > medianChanges, 1, 0)
                if release == 1:
                    df_joined.to_csv(csv_results, index=False)
                else:
                    df_joined.to_csv(csv_results, mode="a", header=False, index=False)

                release += 1
            except Exception as e:
                print(e)
                # print(hashCurrent)
                missing.append(current_hash)

        print(missing)
