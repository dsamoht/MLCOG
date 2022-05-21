import numpy as np
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler


class MakeSplits:
    def __init__(self, dataset, datatype, seed, directory):
        self.dataset = dataset
        self.datatype = datatype
        self.seed = seed
        self.directory = directory
        self.metadata = pd.read_csv("../DATA/5STUDIES_metadata.tsv", sep="\t", index_col=0, header=0)
        self.pathDataType = {}
        self.pathDataType["metaphlan"] = "../DATA/5STUDIES_n966_metaphlan.pkl"
        self.pathDataType["human"] = "../DATA/5STUDIES_n966_human.pkl"
        #self.pathDataType["BGC"] = "../DATA/BGC_clstr_n966.pkl"
        #self.pathDataType["BRENDA"] = "../DATA/BRENDA_clstr_n966.pkl"
        #self.pathDataType["CAZY"] = "../DATA/CAZY_clstr_n966.pkl"
        #self.pathDataType["COG"] = "../DATA/COG_clstr_n966.pkl"
        #self.pathDataType["MERGEM_IS"] = "../DATA/MERGEM_IS_clstr_n966.pkl"
        #self.pathDataType["MERGEM_RG"] = "../DATA/MERGEM_RG_clstr_n966.pkl"
        #self.pathDataType["GBA"] = "../DATA/5STUDIES_n966_gbaxis.pkl"
        #self.pathDataType["ALL_CLSTR"] = "../DATA/ALL_clstr_n966.pkl"
        #self.pathDataType["fusion"] = "../DATA/5STUDIES_n966_fusion.pkl"

        self.makeTrainTestDatasets()

    def loadPkl(self, path):
        """
        Load a serialized object from disk.
        Parameters:
        -----------
        path: path to the object.

        Returns:
        --------
        the object.
        """

        return pickle.load(open(path, "rb"))

    def keepCommon(self, df, n=2):
        """
        Remove features that are not shared by at least n sample(s).
        Parameters:
        -----------
        df: dataframe to filter.
        n: number of samples (default=2).
        
        Returns:
        --------
        the filtered dataframe.
        """
        data_bin = pd.DataFrame(np.where(df > 0, 1, 0),
                                index=list(df.index.values),
                                columns=list(df.columns.values))
        mat_bled = data_bin.loc[:, data_bin.sum() >= n]

        return df.loc[:, mat_bled.columns.values]

    def dispatch(self):
        """
        Return the complete dataframe used for the learning
        and the corresponding labels.
        
        Returns:
        --------
        out: dataframe.
        y: labels array
        """

        df = self.loadPkl(self.pathDataType[self.datatype])
        names = sorted([i for i in df.index if i in self.metadata.index])
        df = df.reindex(names)

        if self.dataset == "ALL":
            out = df[self.metadata['oa_task'] == 1]
        elif self.dataset == "CRC":
            out = df[self.metadata['study'] == "ZELLER2014"]
        elif self.dataset == "IBD":
            out = df[self.metadata['study'] == "QIN2010"]
        elif self.dataset == "LC":
            out = df[self.metadata['study'] == "QIN2014"]
        elif self.dataset == "OB":
            out = df[self.metadata['study'] == "LECHATELIER2013"]
        elif self.dataset == "T2D":
            out = df[self.metadata['study'] == "QIN2012"]

        y = np.array(self.metadata.loc[df.index, 'diseased'])
        out = self.keepCommon(out)

        return out, y

    def makeTrainTestDatasets(self):
        """
        Produce 10 train-test splits with sklearn's StratifiedKFold
        and save the arrays in the cache directory.
        These arrays are used by the Learning class.
        Save also the name of the features selected by the random forest.
        """
        X, y = self.dispatch()
        features_all = X.columns
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)

        for i, (train_index, test_index) in enumerate(cv.split(X, y)):

            y_train, y_test = y[train_index], y[test_index]

            X = np.array(X)

            X_train, X_test = X[train_index], X[test_index]
            scaler = StandardScaler()
            BasicRF = RandomForestClassifier(n_estimators=100, random_state=1)
            selector = SelectFromModel(estimator=BasicRF, threshold='mean')

            X_train = scaler.fit_transform(X_train)
            X_train = selector.fit_transform(X_train, y_train)

            mask = selector.get_support()
            selected_features = features_all[mask]

            X_test = scaler.transform(X_test)
            X_test = selector.transform(X_test)

            PATH = os.getcwd()
            try:
                os.mkdir(PATH + f"{self.directory}")
            except BaseException:
                pass

            with open(f"SELECTED_FEATURES/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_FEATURES_NAMES.pkl", "wb") as f0:
                pickle.dump(selected_features, f0)
            with open(f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_XTRAIN.pkl", "wb") as f1:
                pickle.dump(X_train, f1)
            with open(f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_XTEST.pkl", "wb") as f2:
                pickle.dump(X_test, f2)
            with open(f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_YTRAIN.pkl", "wb") as f3:
                pickle.dump(y_train, f3)
            with open(f"{self.directory}/dataset-{self.dataset}_datatype-{self.datatype}_seed-{self.seed}_iter-{i}_YTEST.pkl", "wb") as f4:
                pickle.dump(y_test, f4)