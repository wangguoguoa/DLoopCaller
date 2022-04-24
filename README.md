# Introduction
DLoopCaller: A deep learning approach for predicting genome-wide chromatin loops by integrating open chromatin landscapes
## Installation
DLoopCaller requires Python3 and several scientific packages to run. It is best to set up a conda environment then install from github. Copy and paste the command snippets below:  
conda config --add channels defaults  
conda config --add channels bioconda  
conda config --add channels conda-forge  
conda create -n DLoopCaller python=3.6 scikit-learn=0.20.2 numpy scipy pandas h5py cooler  
source activate DLoopCaller  
pip install hic-straw  
git clone https://github.com/tariks/peakachu  
## Usage
### Data preparation
DLoopCAller requires the contact map to be a .cool file or a .hic file and any training input to be a text file in bedpe format.   
Training data can be found at my paper data preparation . Cooler files may be found at the 4DN data portal. 
### Generate positive and negative samples

```
python gendata.py -p file.cool -b training.bedpe -a accessible .bigWig -o ./data/
```
### Training model
 
```
python train.py -d ./training data/ -g 0 -b 128 -lr 0.001 -e 30 -w 0.0005 -c ./models
```
### Score

```
python score_chromosome.py -p file.cool -o ./scores -m ./model/chr.pth -a accessible.bigWig
```
### Question
If you have any questions, please send an email to siguo_wang@163.com
