# Install
1. install virtualenv
   - sudo apt-get install python3-pip
   - sudo pip3 install virtualenv
2. clone cpmp
   - git clone github.com/paulorfarah/cpmp
3. install dependencies
   - cd cpmp
   - virtualenv venv
   - source venv/bin/activate
   - pip install pandas pydriller myers matplotlib tensorflow imblearn

#Steps
1. Generate a list of hashes of the releases
   - cd 1.ck
   - git clone <project_url>
   - rename project in ck.py
   - python3 ck.py 
   
2. Run ck.sh with project repository path
   - - ./ck.sh <project_url>
3. Run understand.py with the project name
4. run evolutionary metrics with the methods extracted
   1. it automatically extracts methods
5. run change distiller metrics with the methods extracted
   1. it automatically extracts methods


extra 
4. Extract methods from classes
   1. copy release's hash list to extract_methods of the project
   2. run ./methods.sh <repo_url>
   3. It will generate results in folder results/project_name