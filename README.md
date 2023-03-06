
#Steps
1. Generate a list of hashes of the releases
2. Run ck.sh with project repository path
3. Run understand.py with the project name
4. Extract methods from classes
   1. copy release's hash list to extract_methods of the project
   2. run ./methods.sh <repo_url>
   3. It will generate results in folder results/project_name
5. run evolutionary metrics with the methods extracted
6. run change distiller metrics with the methods extracted
7. 