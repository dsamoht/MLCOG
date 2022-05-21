import argparse
from MakeSplits import MakeSplits
from Learning import Learning


def argparse_command():

    parser = argparse.ArgumentParser(description='''main classification pipeline''')
    parser.add_argument('-s', '--seed', help='''cv's random_state''')
    parser.add_argument('-ds', '--dataset', help='''{ALL, CRC, IBD, LC, OB, T2D}''')
    parser.add_argument('-dt', '--datatype', help='''{ALL_CLSTR, BGC, BRENDA, CAZY, COG, GBA, MERGEM_IS, MERGEM_RG, fusion, human, metaphlan}''')
    parser.add_argument('-t', '--tmpdir', help='''cache directory for splits''')
    parser.add_argument('-a', '--algo', help='''algorithm''')
    parser.add_argument('-sm', '--save_model', action="store_true", help='''save model on disk (in ./SAVED_MODELS/)''')

    return parser.parse_args()


ds = argparse_command().dataset
dt = argparse_command().datatype
tmpdir = argparse_command().tmpdir

if argparse_command().algo:
    algo = argparse_command().algo

else:
    algo = False

if argparse_command().save_model:
    save_models = True

else:
    save_models = False

if argparse_command().seed:
    
    seed = int(argparse_command().seed)
    splitter = MakeSplits(ds, dt, seed, tmpdir)
    Learning(ds, dt, seed, tmpdir, algo, saveModel=save_models)

else:

    for seed in [i*i for i in range(10)]:
        splitter = MakeSplits(ds, dt, seed, tmpdir)
        Learning(ds, dt, seed, tmpdir, algo, saveModel=save_models)