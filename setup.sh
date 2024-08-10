!/bin/bash

# 1. install python 3.12
# 2. create venv 
#    python3 -m venv --system-site-packages venv
# 3. activate venv
source venv/bin/activate
# 4. install jupyter nootbook
#    python3 -m pip install jupyter
# 5. install kernal and setup up a kernal that use venv
#    pip install ipykernel
#    python3 -m ipykernel install --user --name=myenv --display-name="Python (myenv)"
# 6. start jupyter nootbook
python3 -m jupyter notebook
# 7. select the kernal that created in jupyter notebook pages

