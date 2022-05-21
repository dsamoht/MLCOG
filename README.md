# Reference-free microbiome representations enhances host phenotype classification
The repository contains the ML protocol and data used in the [article](). Because gene cluster matrices exceed the size limit of Github, they are not included in the repository. To produce the matrices with your data, see the [protocol](/doc/geneclusters.md).
## Description

* __main.py__ : command line script.
* __MakeSplits.py__ : produce 100 differents train/test splits and save them in a given directory.
* __Learning.py__: learning protocol with hyperparameters tuning and model performance evaluation.


## Dependencies
* Python3
* numpy, pandas, pickle
* [scikit-learn](https://scikit-learn.org/stable/)
* [pyscm](https://github.com/aldro61/pyscm)
* [randomscm](https://github.com/thibgo/randomscm)
## Usage
#### *on a computer cluster
```
python3 main.py -ds T2D -dt metaphlan -t $SLURM_TMPDIR
```
## Help
To see the different options available:
```
python3 main.py --help
```
## Authors
* [Thomas Deschênes](https://github.com/dsamoht)
* [Frédéric Raymond](https://github.com/fredericraymond)
