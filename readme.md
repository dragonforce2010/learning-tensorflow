# Environment setup
## install python
```
brew install python
pip3 install virtualenv
```
## install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
echo 'source /Users/michael.zhang/miniconda3/bin/activate' >> ~/.bash_profile
source ~/.bash_profile
```
## create/activate/deactive a python env
```
conda create -n tensorflow python=3.6
source activate tensorflow
conda deactivate
```

## install the necessary packages like tensorflow, numpy etc.
```
pip install -r requirements.txt
```

## update the requirements.txt when new package installed
```
pip freeze > requirements.txt
```