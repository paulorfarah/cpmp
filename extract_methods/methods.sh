#/bin/bash

check_parameter () {
            : ${1?Repo url required.}
    }
check_parameter $1

REPO=$1
PROJECT_NAME=${REPO##*/}
filename="${PROJECT_NAME}.csv"
mkdir results/"${PROJECT_NAME}"

git clone "$REPO" "$PROJECT_NAME"

while read commit; do
# reading each commit
echo "$commit"
cd "${PROJECT_NAME}"
git checkout $commit
cd ..

java -jar MethodsParser-0.0.1-SNAPSHOT-jar-with-dependencies.jar "${PROJECT_NAME}" "${commit}"
done < $filename

rm -Rf "${PROJECT_NAME}"