"""Command line script."""
import argparse

from mlcog import Learning


if __name__ == "__main__":

    def argparse_command():
        """
        Command line entry.
        """
        parser = argparse.ArgumentParser(description='''main classification pipeline''')
        parser.add_argument('-d', '--datatype', help='''{}''')
        parser.add_argument('-t', '--task', help='''the column name in the labels file''')
        parser.add_argument('-a', '--algo', help='''algorithm''')
        parser.add_argument('-c', '--cache_dir', help='''cache directory for splits''')
        parser.add_argument('-r', '--res_dir', help='''model type for feature selection {}''')
        parser.add_argument('-f', '--feature_selection', help='''selection type {}''')


        return parser.parse_args()

    dt = argparse_command().datatype
    tmpdir = argparse_command().cache_dir
    task = argparse_command().task
    results_dir = argparse_command().res_dir
    fs_model_type = argparse_command().feature_selection
    algo = argparse_command().algo

    Learning(datatype=dt,
                 task=task,
                 algo=algo,
                 cache_dir=tmpdir,
                 results_dir=results_dir,
                 selection_model_type=fs_model_type)
