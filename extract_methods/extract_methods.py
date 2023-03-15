import argparse
import os
import subprocess

import pydriller as pydriller

def check(chash):
    if os.path.exists(chash + '.csv'):
        print("Commit", chash, "already collected, skipping...")
        return True
    else:
        print("Commit", chash, "not found, collecting...")
        return False

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Extractor for changeDistiller')
    ap.add_argument('--project', required=True)
    args = ap.parse_args()
    with open('commits-' + args.project + '.csv') as f:
        commits = f.read().splitlines()
    gr = pydriller.Git(args.project)

    for hash in commits:
        if check(hash):
            continue

        gr.clear()
        print("git checkout on commit", hash + "...")
        gr.checkout(hash)

        print("creating the project", hash + ".udb ...")
        subprocess.run(['java', '-jar', 'MethodsParser-0.0.1-SNAPSHOT.jar', args.project, hash])
