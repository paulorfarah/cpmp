#/bin/bash

check_parameter () {
	    : ${1?Repo url required.}
    }
check_parameter $1

# set target directories
DIR_A="projectA/"
DIR_B="projectB/"

# clean target directories
rm -Rf "$DIR_A"
rm -Rf "$DIR_B"

#clone repository to target directories
REPO=$1
PROJECT_NAME=${REPO##*/}
DIR_A="${DIR_A}${PROJECT_NAME}"
DIR_B="${DIR_B}${PROJECT_NAME}"

git clone "$REPO" "$DIR_A"
git clone "$REPO" "$DIR_B"
#echo "Script executed from: ${PWD}"
nohup python3 changeDistiller.py --pathA "${DIR_A}" --pathB "${DIR_B}" --projectName "${PROJECT_NAME}" --absolutePath "${PWD}/" --mode "tag" &
