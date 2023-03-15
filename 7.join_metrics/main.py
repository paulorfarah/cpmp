from asyncore import write
from importlib.resources import path
import pydriller
import argparse
from csv import reader
import csv
from pydriller import Repository
import git
import pandas as pd
import numpy as np

from ck import join_ck
from understand import join_understand
from change_distiller import join_change_distiller

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Join Metrics')
    # args = ap.parse_args()

    projects = ['commons-bcel']
    for project_name in projects:
        repo_path = "../repos/" + project_name
        gr = pydriller.Git(repo_path)
        repo = git.Repo(repo_path)
        tags = repo.tags
        release = 1

        csvOrganic = "/Volumes/backup-geni/projects-smells/results/organic/junit4.csv"
        csvResults = "/Volumes/backup-geni/projects-smells/results/junit4-all-releases.csv"

        organicMetrics = ["projectName", "commitNumber", "fullyQualifiedName",
                          "PublicFieldCount", "IsAbstract", "ClassLinesOfCode", "WeighOfClass",
                          "FANIN", "TightClassCohesion", "FANOUT", "OverrideRatio", "LCOM3",
                          "WeightedMethodCount", "LCOM2", "NumberOfAccessorMethods",
                          'LazyClass', 'DataClass', 'ComplexClass', 'SpaghettiCode',
                          'SpeculativeGenerality', 'GodClass', 'RefusedBequest',
                          'ClassDataShouldBePrivate', 'BrainClass', 'TotalClass',
                          'LongParameterList', 'LongMethod', 'FeatureEnvy',
                          'DispersedCoupling', 'MessageChain', 'IntensiveCoupling',
                          'ShotgunSurgery', 'BrainMethod', 'TotalMethod', 'TotalClassMethod',
                          "DiversityTotal", "DiversityMethod", "DiversityClass"]
        # f = open(csvPath, "w")
        # writer = csv.writer(f)
        missing = []
        for tag in tags:
            current_hash = gr.get_commit_from_tag(tag.name).hash

            try:
                releaseOrganicMetrics = pd.read_csv(csvOrganic, usecols=organicMetrics, sep=',', engine='python',
                                                    index_col=False)
                releaseOrganicMetrics = releaseOrganicMetrics[(releaseOrganicMetrics['commitNumber'] == current_hash)]

                print("Organic ")
                print(releaseOrganicMetrics.shape[0])

                releaseCK = join_ck(project_name, current_hash)
                releaseUnderstand = join_understand(project_name, current_hash)
                df_joined = pd.merge(left=releaseCK, right=releaseUnderstand, left_on='class', right_on='Name')


                # df_joined = pd.merge(left=df_joined, right=releaseProcessMetrics, left_on='class',
                #                                  right_on='className')
                # df_joined = pd.merge(left=df_joined, right=releaseOrganicMetrics,
                #                                          left_on='class', right_on='fullyQualifiedName')

                # merged_full = pd.merge(left=ck_understand_process_organic, right=releaseChangeDistillerMetrics,
                #                        left_on='class', right_on='CLASS_PREVIOUSCOMMIT')

                release_change_distiller = join_change_distiller(project_name, current_hash)
                df_joined = pd.merge(left=df_joined, right=release_change_distiller,
                                     left_on='class', right_on='CLASS_PREVIOUSCOMMIT')
                df_joined = join_change_distiller(project_name, current_hash, df_joined)

                # df_joined.loc[:,'class_frequency'] = 1
                df_joined.loc[:, 'will_change'] = 0
                # df_joined.loc[:,'number_of_changes'] = 0
                df_joined.loc[:, 'release'] = release
                medianChanges = df_joined['TOTAL_CHANGES'].median()
                df_joined['will_change'] = np.where(df_joined['TOTAL_CHANGES'] > medianChanges, 1, 0)
                if (release == 1):
                    df_joined.to_csv(csvResults, index=False)
                else:
                    df_joined.to_csv(csvResults, mode="a", header=False, index=False)

                release += 1
            except Exception as e:
                print(e)
                # print(hashCurrent)
                missing.append(current_hash)

        print(missing)
