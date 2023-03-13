import filecmp
import os
import subprocess
import sys
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


# def count_metrics_between_releases(new_file, old_file h_prev):
#     previous_commit = None
#     for current_commit in commits_list:
#         if not previous_commit:
#             # first commit
#             previous_commit = current_commit
#         else:
#             path_curr.checkout(current_commit)
#             path_prev.checkout(current_commit)
#

def read_method_files(hash, file, methods_path):
    subprocess.call([args.java, '-jar',
                     'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file,
                     hash])
    method_files = None
    try:
        method_files = [f for f in listdir(methods_path) if isfile(join(methods_path, f))]
    except FileNotFoundError:
        print('[>>>FileNotFound]: Check if class has methods:')
        if methods_path:
            print(methods_path)
        print('----------------------------------------------')
    except:
        print('[>>>Error]: extracting methods: ' + str(sys.exc_info()))
    return method_files

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extract process metrics')
    ap.add_argument('--pathA', required=True)
    ap.add_argument('--pathB', required=True)
    ap.add_argument('--project_name', required=True)
    ap.add_argument('--java', required=True)
    args = ap.parse_args()

    # folder with repo: projectA and projectB

    path_A = pydriller.Git(args.pathA)
    path_B = pydriller.Git(args.pathB)

    #tags
    repo = git.Repo(args.pathA)
    tags = repo.tags

    release = 1
    commit_A = ''
    commit_B = ''
    boc_list = {}
    fch_list = {}
    frchArray = {}
    wcdArray = {}
    wfrArray = {}
    lcaArray = {}
    lcdArray = {}
    csbArray = {}
    csbsArray = {}
    acdfArray = {}

    absolute_path = os.getcwd() + '/'
    csv_path = absolute_path + "results/" + args.project_name + "-methods-results-processMetrics.csv"
    f = open(csv_path, "w")
    writer = csv.writer(f)
    for tag in tags:
        hash_current = path_A.get_commit_from_tag(tag.name).hash
        path_A.checkout(hash_current)
        if commit_B == '':
            hash_previous = None
        filesA = path_A.files()
        filesA = [x for x in filesA if x.endswith('.java')]

        if release == 1:
            commit_A = tag
            row = ['project', 'commit', 'commitprevious', 'class', 'release', 'BOC', 'TACH', 'FCH', 'LCH', 'CHO',
                   'FRCH', 'CHD', 'WCD', 'WFR', 'ATAF', 'LCA', 'LCD', 'CSB', 'CSBS', 'ACDF']
            writer.writerow(row)
            for file in filesA:
                # '/usr/lib/jvm/java-19-openjdk-amd64/bin/java'
                methods_path_A = 'results/' + hash_current + file
                # subprocess.call([args.java, '-jar',
                #                  'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file,
                #                  hash_current])
                # method_files_A = None
                # try:
                #     method_files_A = [f for f in listdir(methods_path_A) if isfile(join(methods_path_A, f))]
                # except FileNotFoundError:
                #     if method_files_A:
                #         print('[>>>FileNotFound]: ' + methods_path_A + '! \n Check if class has methods...')
                method_files_A = read_method_files(hash_current, file, methods_path_A)
                if method_files_A:
                    for method_file in method_files_A:
                        method_file_A = methods_path_A + '/' + method_file
                        method_file_A_renamed = absolute_path + transform_method_to_class(method_file_A)
                        method_short_name = method_file_A_renamed.split(args.pathA, 1)[1]
                        if (method_short_name not in boc_list):
                            boc_list[method_short_name] = release
                            fch_list[method_short_name] = 0

        else:
            # release > 1
            project = args.project_name
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
            hash_previous = path_A.get_commit_from_tag(commit_A.name).hash

            for file in filesA:
                methods_path_A = 'results/' + hash_current + file
                method_files_A = read_method_files(hash_current, file, methods_path_A)

                if method_files_A:
                    for method_file in method_files_A:
                        method_file_A = methods_path_A + '/' + method_file

                        #need to transform? I need to check, I don't think so...
                        method_file_A_renamed = absolute_path + transform_method_to_class(method_file_A)

                        method_short_name = method_file_A_renamed.split(args.pathA, 1)[1]
                        if method_short_name not in boc_list:
                            boc_list[method_short_name] = release
                            boc = release
                        else:
                            boc = boc_list.get(method_short_name)
                        if method_short_name not in fch_list:
                            fch_list[method_short_name] = 0
                        if method_short_name not in frchArray:
                            frchArray[method_short_name] = 0
                        if method_short_name not in wcdArray:
                            wcdArray[method_short_name] = 0
                        if method_short_name not in wfrArray:
                            wfrArray[method_short_name] = 0
                        if method_short_name not in lcaArray:
                            lcaArray[method_short_name] = 0
                        if method_short_name not in lcdArray:
                            lcdArray[method_short_name] = 0
                        if method_short_name not in csbArray:
                            csbArray[method_short_name] = 0
                        if method_short_name not in csbsArray:
                            csbsArray[method_short_name] = 0
                        if method_short_name not in acdfArray:
                            acdfArray[method_short_name] = 0


                        # get all commits from release n-1 to n, the goal is to find the total amount of changes on a file
                        commits_touching_path = Repository(args.pathA, from_commit=hash_previous,
                                                           to_commit=hash_current).traverse_commits()
                        # file_temp = method_short_name.replace(
                        #     absolutePath + args.pathA + args.project_name + "/", '')
                        added_lines = 0
                        removed_lines = 0
                        loc = 0
                        previous_commit = None
                        for current_commit in commits_touching_path:
                            path_A.checkout(current_commit.hash)
                            if previous_commit:
                                path_B.checkout(previous_commit.hash)
                            if not previous_commit:
                                # first commit
                                previous_commit = current_commit
                            else:
                                current_modified_files = [x for x in current_commit.modified_files if x.filename.endswith('.java')]
                                # previous_files = pathB.files()

                                for modified_file in current_modified_files:
                                    # print(modified_file.filename)

                                    if modified_file.new_path and modified_file.old_path:
                                        print(modified_file.new_path + ' ' + modified_file.old_path)



                                        commit_methods_path_A = 'results/' + current_commit.hash + modified_file.new_path
                                        commit_method_files_A = read_method_files(current_commit.hash, modified_file.new_path, commit_methods_path_A)

                                        commit_methods_path_B = 'results/' + previous_commit.hash + modified_file.old_path
                                        commit_method_files_B = read_method_files(previous_commit.hash, modified_file.old_path, commit_methods_path_B)

                                        result = filecmp.dircmp(methods_path_A, commit_methods_path_B)
                                        print(result.report())

                                        #new methods
                                        for new_method in result.left_only:
                                            print(new_method)
                                            try:
                                                with open(commit_methods_path_A + '/' + new_method, 'r') as fp:
                                                    lines = len(fp.readlines())
                                                    commit_method_file_A_renamed = commit_methods_path_A + '/' + new_method
                                                    commit_method_short_name = \
                                                    commit_method_file_A_renamed.split(args.pathA, 1)[1]
                                                    csbsArray[method_short_name] = lines
                                            except:
                                                print(new_method)
                                                print(sys.exc_info())
                                        #modified methods
                                        for modified_method in result.diff_files:
                                            print(modified_method)

                                        # 'results/' + hash_current + file
                                    # elif modified_file.old_path:
                                    #     print('old: ' + modified_file.old_path)
                                    elif modified_file.new_path:
                                        # print('new: ' + modified_file.new_path)
                                        current_commit_hash = current_commit.hash
                                        current_commit_path = str(path_A.path) + '/' + str(modified_file.new_path)
                                        commit_methods_path_A = 'results/' + current_commit.hash + str(path_A.path) + '/' + modified_file.new_path

                                        commit_method_files_A = read_method_files(current_commit_hash, current_commit_path, commit_methods_path_A)


                                        # commit_method_files_A = read_method_files(current_commit.hash, modified_file.new_path,
                                        #                                    commit_methods_path_A)
                                        if commit_method_files_A:
                                            for commit_method_file_A in commit_method_files_A:
                                                try:
                                                    with open(commit_methods_path_A + '/' + commit_method_file_A, 'r') as fp:
                                                        lines = len(fp.readlines())
                                                        commit_method_file_A_renamed = commit_methods_path_A + '/' + commit_method_file_A
                                                        commit_method_short_name = commit_method_file_A_renamed.split(args.pathA, 1)[1]
                                                        csbsArray[method_short_name] = lines
                                                except:
                                                    print(commit_method_file_A)
                                                    print(sys.exc_info())

                                #     print('old: None')
                                # print('-------------')
                                # count_metrics_between_releases(modified_file.new_path, modified_file.old_path, pathA, pathB )
                                # if (modified_file.change_type.name == 'ADD' and (
                                #         modified_file.new_path == modified_file or modified_file.old_path == modified_file)):
                                #     # size of
                                #     csbsArray[method_short_name] = modified_file.nloc
                                # if (modified_file.change_type.name == 'MODIFY' and (
                                #         modified_file.new_path == modified_file or modified_file.old_path == modified_file)):


                                    # loc = modified_file.nloc
                                    # added_lines += modified_file.added_lines
                                    # removed_lines += modified_file.deleted_lines
                                    # # first time change
                                    # if (fchArray[method_short_name] == 0):
                                    #     fchArray[method_short_name] = release
                                    #     fch = release
                                    # # last time change, the lastest released analayzed
                                    # lhc = release
                                    # # if changes have occurred
                                    # cho = 1
                                    # # frequency of change
                                    # frchArray[method_short_name] += 1
                                    # frch = frchArray[method_short_name]

                        # total amount change, added lines + deleted lines (changed lines are already counted twice )
                        tach = added_lines + removed_lines
                        if (tach > 0):
                            chd = tach / loc
                            # cumulative weight of change
                            wcdArray[method_short_name] += tach * pow(2, boc - release)
                            # sum of change density, to normalize later
                            acdfArray[method_short_name] += chd
                            # agregate change size, normalized by frequency change
                            if (frch > 0):
                                ataf = tach / frch
                                # agregate change density normalized by frch
                                acdf = acdfArray[method_short_name] / frch
                            # last amount of change
                            lcaArray[method_short_name] = tach
                            # last change density
                            lcdArray[method_short_name] = chd
                            csbArray[method_short_name] += tach

                        wcd = wcdArray[method_short_name]
                        wch = wcd * pow(2, boc - release)
                        # cumultive weight frequecy
                        wfrArray[method_short_name] += (release - 1) * cho
                        wfr = wfrArray[method_short_name]
                        lca = lcaArray[method_short_name]
                        lcd = lcdArray[method_short_name]
                        csb = csbArray[method_short_name]
                        if (csb > 0):
                            csbs = csbsArray[method_short_name] / csb

                        row = [args.project_name, hash_current, hash_previous, method_short_name, release, boc,
                               tach, fch, lch, cho, frch,
                               chd, wch, wfr, ataf, lca, lcd, csb, csbs, acdf]
                        writer.writerow(row)

        commit_A = tag
        release += 1
    f.close()
