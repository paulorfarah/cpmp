import pydriller
import argparse
from csv import reader
import shutil
import subprocess

import git
from os import listdir
from os.path import isfile, join


def transform_method_to_class(method_file):
    code = []
    file_name = method_file.replace('(', '_').replace(')', '_').replace(' ', '').replace(',', '_').replace('[',
                                                                                                           '_').replace(
        ']', '')

    class_sig = 'public class Fake_function_qwerty {'
    with open(method_file, 'r') as f:
        code = f.readlines()
        if not code[0].startswith(class_sig):
            code.insert(0, class_sig)
            code.append('}')
    with open(file_name, 'w') as fw:
        fw.writelines(code)
    return file_name


def compare_classes(pathA, pathB, currentCommit, previousCommit):
    filesA = pathA.files()
    filesB = pathB.files()
    filesA = [x for x in filesA if x.endswith('.java')]
    filesB = [x for x in filesB if x.endswith('.java')]
    csvPath = args.absolutePath + 'results/' + args.projectName + "-methods-results.csv"
    try:
        f = open(csvPath, "x")
    except:
        print("file exists")
    for file in filesA:
        file_temp = file.replace(args.absolutePath + "projectA", '')
        if any(file_temp in s for s in filesB):
            file2 = args.absolutePath + "projectB" + file_temp
            # classPreviousCommit classCurrentCommit csvPath projectName currentCommit previousCommit

            subprocess.call(['/usr/lib/jvm/java-19-openjdk-amd64/bin/java', '-jar',
                             'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file, currentCommit])
            subprocess.call(['/usr/lib/jvm/java-19-openjdk-amd64/bin/java', '-jar',
                             'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file2,
                             previousCommit])

            methods_path_A = 'results/' + currentCommit + file
            methods_path_B = 'results/' + previousCommit + file2
            method_files_A = None
            method_files_B = None
            try:
                method_files_A = [f for f in listdir(methods_path_A) if isfile(join(methods_path_A, f))]
            except FileNotFoundError:
                if method_files_A:
                    print('[>>>FileNotFound]: ' + methods_path_A + '! \n Check if class has methods...')
            try:
                method_files_B = [f for f in listdir(methods_path_B) if isfile(join(methods_path_B, f))]
            except FileNotFoundError:
                if method_files_B:
                    print('[>>>FileNotFound]: ' + methods_path_B + '! \n Check if class has methods...')

            if method_files_A and method_files_B:
                for method_file in method_files_A:
                    if any(method_file == f for f in method_files_B):
                        method_file_A = methods_path_A + '/' + method_file
                        method_file_B = methods_path_B + '/' + method_file

                        method_file_A_renamed = args.absolutePath + transform_method_to_class(method_file_A)
                        method_file_B_renamed = args.absolutePath + transform_method_to_class(method_file_B)
                        cd_cmd = '/usr/lib/jvm/java-8-openjdk-amd64/bin/java -jar ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar ' + method_file_A_renamed + ' ' + method_file_B_renamed + ' ' + csvPath + ' ' + args.projectName + ' ' + currentCommit + ' ' + previousCommit
                        print(cd_cmd)
                        # subprocess.call(['java', '-jar', 'ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar',
                        #                  '"' + method_file_B + '"', '"' + method_file_A + '"', csvPath, args.projectName, currentCommit, previousCommit])
                        print(subprocess.check_output((['/usr/lib/jvm/java-8-openjdk-amd64/bin/java', '-jar',
                                                        'ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar',
                                                        method_file_A_renamed, method_file_B_renamed, csvPath,
                                                        args.projectName, currentCommit, previousCommit])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extractor for changeDistiller')
    ap.add_argument('--pathA', required=True)
    ap.add_argument('--pathB', required=True)
    ap.add_argument('--projectName', required=True)
    ap.add_argument('--absolutePath', required=True)
    args = ap.parse_args()

    pathA = pydriller.Git(args.pathA)
    pathB = pydriller.Git(args.pathB)
    repo = git.Repo(args.pathA)
    tags = repo.tags

    i = 0
    commit_A = ''
    commit_B = ''
    for tag in tags:
        if i == 0:
            commit_A = tag
            i += 1
        else:
            hashA = pathA.get_commit_from_tag(commit_A.name).hash
            hashB = pathB.get_commit_from_tag(tag.name).hash
            pathA.checkout(hashA)
            pathB.checkout(hashB)
            compare_classes(pathA, pathB, str(hashA), str(hashB))
            commit_A = tag
