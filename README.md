# Gene-based microbiome representation enhances host phenotype classification
The repository contains the ML protocol and data used in the [article](https://github.com/dsamoht/MLCOG). Because the majority of matrices exceed the size limit of Github, they are not included in the repository. To produce the gene clusters matrices with your data, see the [protocol](/doc/geneclusters.md).


## Dependencies
* numpy, pandas, pickle
* [scikit-learn](https://scikit-learn.org/stable/)
* [pyscm](https://github.com/aldro61/pyscm)
* [randomscm](https://github.com/thibgo/randomscm)
* [xgboost](https://pypi.org/project/xgboost/)

## Usage
*on a computer cluster
```
python3 main.py -a MNB -d metaphlan -c $SLURM_TMPDIR -t GlucF_SMET  -r results_dir -f elastic_net
```
## Help
To see the different options available:
```
python3 main.py --help
```
## Authors
* [Thomas Deschênes](https://github.com/dsamoht)
* [Frédéric Raymond](https://github.com/fredericraymond)
