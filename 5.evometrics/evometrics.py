import os
import subprocess
from asyncore import write
from os import listdir
from os.path import isfile, join

import pydriller
import argparse
from csv import reader
import csv
from pydriller import Repository
import git


def transform_method_to_class(method_file):
    code = []
    file_name = method_file.replace('(', '_').replace(')', '_').replace(' ', '').replace(',', '_').replace('[', '_')\
        .replace(']', '')

    class_sig = 'public class Fake_function_qwerty {'
    with open(method_file, 'r') as f:
        code = f.readlines()
        if not code[0].startswith(class_sig):
            code.insert(0, class_sig)
            code.append('}')
    with open(file_name, 'w') as fw:
        fw.writelines(code)
    return file_name


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extract process metrics')
    ap.add_argument('--pathA', required=True)
    ap.add_argument('--projectName', required=True)
    args = ap.parse_args()

    # folder with repo: projectA and projectB

    pathA = pydriller.Git(args.pathA)
    repo = git.Repo(args.pathA)
    tags = repo.tags

    release = 1
    commit_A = ''
    commit_B = ''
    bocArray = {}
    fchArray = {}
    frchArray = {}
    wcdArray = {}
    wfrArray = {}
    lcaArray = {}
    lcdArray = {}
    csbArray = {}
    csbsArray = {}
    acdfArray = {}

    absolutePath = os.getcwd() + '/'
    csvPath = absolutePath + "results/" + args.projectName + "-methods-results-processMetrics.csv"
    f = open(csvPath, "w")
    writer = csv.writer(f)
    for tag in tags:
        hashCurrent = pathA.get_commit_from_tag(tag.name).hash
        pathA.checkout(hashCurrent)
        if (commit_B == ''):
            hashPrevious = None
        filesA = pathA.files()
        filesA = [x for x in filesA if x.endswith('.java')]

        if (release == 1):
            commit_A = tag
            row = ['project', 'commit', 'commitprevious', 'class', 'release', 'BOC', 'TACH', 'FCH', 'LCH', 'CHO',
                   'FRCH', 'CHD', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS', 'ACDF']
            writer.writerow(row)
            for file in filesA:
                subprocess.call(['/usr/lib/jvm/java-19-openjdk-amd64/bin/java', '-jar',
                                 'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file,
                                 hashCurrent])

                methods_path_A = 'results/' + hashCurrent + file
                method_files_A = None
                try:
                    method_files_A = [f for f in listdir(methods_path_A) if isfile(join(methods_path_A, f))]
                except FileNotFoundError:
                    if method_files_A:
                        print('[>>>FileNotFound]: ' + methods_path_A + '! \n Check if class has methods...')

                if method_files_A:
                    for method_file in method_files_A:
                        method_file_A = methods_path_A + '/' + method_file
                        method_file_A_renamed = absolutePath + transform_method_to_class(method_file_A)
                        if (method_file_A_renamed not in bocArray):
                            bocArray[method_file_A_renamed] = release
                            fchArray[method_file_A_renamed] = 0

        else:
            project = args.projectName
            commit = hashCurrent
            commitprevious = hashPrevious
            boc = release
            tach = 0
            fch = 0
            lch = release
            cho = 0
            frch = 0
            chd = 0
            wcd = 0
            wfr = 0
            ataf = 0
            lca = 0
            lcd = 0
            csb = 0
            csbs = 0
            acdf = 0
            hashPrevious = pathA.get_commit_from_tag(commit_A.name).hash

            for file in filesA:
                subprocess.call(['/usr/lib/jvm/java-19-openjdk-amd64/bin/java', '-jar',
                                 'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file,
                                 hashCurrent])

                methods_path_A = 'results/' + hashCurrent + file
                method_files_A = None
                try:
                    method_files_A = [f for f in listdir(methods_path_A) if isfile(join(methods_path_A, f))]
                except FileNotFoundError:
                    if method_files_A:
                        print('[>>>FileNotFound]: ' + methods_path_A + '! \n Check if class has methods...')

                if method_files_A:
                    for method_file in method_files_A:
                        method_file_A = methods_path_A + '/' + method_file
                        method_file_A_renamed = absolutePath + transform_method_to_class(method_file_A)
                        if (method_file_A_renamed not in bocArray):
                            bocArray[method_file_A_renamed] = release
                            fchArray[method_file_A_renamed] = 0

                            if (method_file_A_renamed not in bocArray):
                                bocArray[method_file_A_renamed] = release
                                boc = release
                            else:
                                boc = bocArray.get(method_file_A_renamed)
                            if (method_file_A_renamed not in fchArray):
                                fchArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in frchArray):
                                frchArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in wcdArray):
                                wcdArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in wfrArray):
                                wfrArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in lcaArray):
                                lcaArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in lcdArray):
                                lcdArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in csbArray):
                                csbArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in csbsArray):
                                csbsArray[method_file_A_renamed] = 0
                            if (method_file_A_renamed not in acdfArray):
                                acdfArray[method_file_A_renamed] = 0
                            # get all commits from release n-1 to n, the goal is to find the total amount of changes on a file
                            commits_touching_path = Repository(args.pathA, from_commit=hashPrevious,
                                                               to_commit=hashCurrent).traverse_commits()
                            file_temp = method_file_A_renamed.replace(
                                absolutePath + "projectA/" + args.projectName + "/", '')
                            added_lines = 0
                            removed_lines = 0
                            loc = 0
                            for cc in commits_touching_path:
                                modifiedFiles = [x for x in cc.modified_files if x.filename.endswith('.java')]
                                for m in modifiedFiles:
                                    if (m.change_type.name == 'ADD' and (
                                            m.new_path == file_temp or m.old_path == file_temp)):
                                        # size of
                                        csbsArray[method_file_A_renamed] = m.nloc
                                    if (m.change_type.name == 'MODIFY' and (
                                            m.new_path == file_temp or m.old_path == file_temp)):
                                        loc = m.nloc
                                        added_lines += m.added_lines
                                        removed_lines += m.deleted_lines
                                        # first time change
                                        if (fchArray[method_file_A_renamed] == 0):
                                            fchArray[method_file_A_renamed] = release
                                            fch = release
                                        # last time change, the lastest released analayzed
                                        lhc = release
                                        # if changes have occurred
                                        cho = 1
                                        # frequency of change
                                        frchArray[method_file_A_renamed] += 1
                                        frch = frchArray[method_file_A_renamed]

                            # total amount change, added lines + deleted lines (changed lines are already counted twice )
                            tach = added_lines + removed_lines
                            if (tach > 0):
                                chd = tach / loc
                                # cumulative weight of change
                                wcdArray[method_file_A_renamed] += tach * pow(2, boc - release)
                                # sum of change density, to normalize later
                                acdfArray[method_file_A_renamed] += chd
                                # agregate change size, normalized by frequency change
                                if (frch > 0):
                                    ataf = tach / frch
                                    # agregate change density normalized by frch
                                    acdf = acdfArray[method_file_A_renamed] / frch
                                # last amount of change
                                lcaArray[method_file_A_renamed] = tach
                                # last change density
                                lcdArray[method_file_A_renamed] = chd
                                csbArray[method_file_A_renamed] += tach

                            wcd = wcdArray[method_file_A_renamed]
                            wch = wcd * pow(2, boc - release)
                            # cumultive weight frequecy
                            wfrArray[method_file_A_renamed] += (release - 1) * cho
                            wfr = wfrArray[method_file_A_renamed]
                            lca = lcaArray[method_file_A_renamed]
                            lcd = lcdArray[method_file_A_renamed]
                            csb = csbArray[method_file_A_renamed]
                            if (csb > 0):
                                csbs = csbsArray[method_file_A_renamed] / csb

                            row = [args.projectName, hashCurrent, hashPrevious, method_file_A_renamed, release, boc,
                                   tach, fch, lch, cho, frch,
                                   chd, wch, wfr, ataf, lca, lcd, csb, csbs, acdf]
                            writer.writerow(row)

        commit_A = tag
        release += 1
    f.close()
