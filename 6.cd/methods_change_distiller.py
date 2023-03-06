import pydriller
import argparse
from csv import reader
import shutil
import subprocess

import git
from os import listdir
from os.path import isfile, join

          
def compare_classes(pathA, pathB, currentCommit, previousCommit):
    filesA = pathA.files()
    filesB = pathB.files()
    filesA = [x for x in filesA if x.endswith('.java')]
    filesB = [x for x in filesB if x.endswith('.java')]
    csvPath = args.absolutePath + args.projectName + "-methods-results.csv"
    try:
        f = open(csvPath, "x")
    except:
        print("file exists")
    for file in filesA:
        file_temp = file.replace(args.absolutePath+"projectA", '')
        if any(file_temp in s for s in filesB):
            file2 = args.absolutePath+"projectB" + file_temp
            #classPreviousCommit classCurrentCommit csvPath projectName currentCommit previousCommit

            subprocess.call(['java', '-jar', 'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file, currentCommit])
            subprocess.call(['java', '-jar', 'JMethodsExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar', 'file', file2, previousCommit])

            methods_path_A = 'results/' + currentCommit + file
            method_files_A = [f for f in listdir(methods_path_A) if isfile(join(methods_path_A, f))]
            methods_path_B = 'results/' + previousCommit + file2
            method_files_B = [f for f in listdir(methods_path_B) if isfile(join(methods_path_B, f))]
            for method_file in method_files_A:
                if any(method_file == f for f in method_files_B):
                    method_file_B = methods_path_B + '/' + method_file
                    method_file_A = methods_path_A + '/' + method_file
                    cd_cmd = 'java -jar ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar "' + method_file_B + '" "' + method_file_A + '" ' + csvPath + ' ' + args.projectName + ' ' + currentCommit + ' ' + previousCommit
                    print(cd_cmd)
                    # subprocess.call(['java', '-jar', 'ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar',
                    #                  '"' + method_file_B + '"', '"' + method_file_A + '"', csvPath, args.projectName, currentCommit, previousCommit])
                    print(subprocess.check_output((['java', '-jar', 'ChangeDistillerReader-0.0.1-SNAPSHOT-jar-with-dependencies.jar',
                                     '"' + method_file_B + '"', '"' + method_file_A + '"', csvPath, args.projectName, currentCommit, previousCommit])))

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
