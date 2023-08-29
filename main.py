"""Command line script."""
import argparse
import shutil

from mlcog import Learning


if __name__ == "__main__":

    def argparse_command():
        """
        Command line entry.
        """
        parser = argparse.ArgumentParser(description='''main classification pipeline''')
        parser.add_argument('-d', '--datatype', help='''datatype name included in the file `datapaths.json`''')
        parser.add_argument('-t', '--task', help='''column name in the metadata file''')
        parser.add_argument('-a', '--algo', help='''algorithm among `SVM_rbf`, `SVM_L1`, `SVM_L2`, `Logistic_Regression_L1`,
                                                    `Logistic_Regression_L2`, `Decision_tree`, `random_forest`,
                                                    `SCM`, `Random_SCM`, `XGBoost` and `MNB`''')
        parser.add_argument('-c', '--cache_dir', help='''/path/to/cache/directory''')
        parser.add_argument('-r', '--res_dir', help='''/path/to/output/directory''')
        parser.add_argument('-f', '--feature_selection', help='''feature selection strategy among `random_forest`, `elastic_net` and `none`''')
        parser.add_argument('--test', action='store_true', help='''test the installation''')

        return parser.parse_args()

    if not argparse_command().test:
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
    else:
        dt, task = "test", "test"
        tmpdir = "./test/tmpdir/"
        results_dir = "./test/test_out/"
        for algo in ['SVM_rbf', 'SVM_L1', 'SVM_L2', 'Logistic_Regression_L1',
                     'Logistic_Regression_L2', 'Decision_tree', 'random_forest',
                     'SCM', 'Random_SCM', 'XGBoost', 'MNB']:
            for selection in ['random_forest', 'elastic_net', 'none']:
                Learning(datatype=dt,
                    task=task,
                    algo=algo,
                    cache_dir=tmpdir,
                    results_dir=results_dir,
                    n_iter=1,
                    n_split=5,
                    selection_model_type=selection)

        shutil.rmtree(tmpdir)
