# Gene-based microbiome representation enhances host phenotype classification
The repository contains the ML protocol and data used in the [article](https://journals.asm.org/doi/10.1128/msystems.00531-23). Because the majority of matrices exceed the size limit of Github, they are not included in the repository. To produce the gene clusters matrices with your data, see the [protocol](/doc/geneclusters.md).

## Data formatting
metadata (rows: samples, columns: tasks):
```
sample,GlucF_SMET,GlucF_PDT2,GlucF_DT2
bl_031v4,0,0,1
bl_014v5,0,1,0
bl_024v5,1,0,0
bl_029v2,0,1,0
bl_022v8,1,0,0
```
matrices (rows: samples, columns: features):
```
sample,k__Bacteria,k__Archaea,k__Bacteria|p__Firmicutes
bl_031v4,100.0,0.0,55.85994
bl_014v5,98.3,1.124,48.9800
bl_024v5,100.0,0.0,64.6853
bl_029v2,100.0,2.170,48.8594
bl_022v8,100.0,1.452,53.2356
```
To save your matrices in pickled dataframes:
```python
import pandas as pd
import pickle

dframe = pd.read_csv("my_matrix.csv", index_col=0, header=0)
with open("my_matrix.pkl", "wb") as pickled_output:
    pickle.dump(dframe, pickled_output)
```

## Edit `datapaths.json`
Include all the data types and their corresponding path.

## Usage
```
python main.py -a MNB -d metaphlan -c $SLURM_TMPDIR -t GlucF_SMET  -r results_dir -f elastic_net
```
*you can change $SLURM_TMPDIR for another cache directory if you are not using a HPC.

## Testing: execute each algorithm on a toy dataset
### (~2 minutes with 8 threads)
```
python main.py --test
```
## Authors
* [Thomas Deschênes](https://github.com/dsamoht)
* [Frédéric Raymond](https://github.com/fredericraymond)
