import pandas as pd
import os

class DataManager:
    _instance = None
    _loadtrain = None
    _loadvalidate = None
    _loadtest = None
    _featuretrain = None
    _featurevalidate = None
    _featuretest = None
    _finaltrain = None
    _finalvalidate = None
    _finaltest = None
    _nn_train = None
    _nn_validate = None
    _nn_test = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __load_loaddata(self):
        if self._loadtrain is None:
            self._loadtrain, self._loadvalidate, self._loadtest = self.__load_fileloaddata()
    def __load_loadfeature(self):
        if self._featuretrain is None:
            self._featuretrain, self._featurevalidate, self._featuretest = self.__load_filefeaturedata()
    def __load_loadfinal(self):
        if self._finaltrain is None:
            self._finaltrain, self._finalvalidate, self._finaltest = self.__load_filefinaldata()
    def __load_nn_data(self):
        if self._nn_train is None:
            self._nn_train, self._nn_validate, self._nn_test = self.__load_filenn_data()

    def __load_fileloaddata(self):
        """Load train, validation, and test datasets"""
        loadtrain = pd.read_csv("data/processed/cleaned_train.csv")
        loadvalidate = pd.read_csv("data/processed/cleaned_validate.csv")
        loadtest = pd.read_csv("data/processed/cleaned_test.csv")
        return loadtrain, loadvalidate, loadtest
    
    def __load_filefeaturedata(self):
        """Load train, validation, and test datasets"""
        featuretrain = pd.read_csv("data/processed/feature_train.csv")
        featurevalidate = pd.read_csv("data/processed/feature_validate.csv")
        featuretest = pd.read_csv("data/processed/feature_test.csv")
        return featuretrain, featurevalidate, featuretest
    
    def __load_filenn_data(self):
        """Load nearest neighbor datasets"""
        nn_train = pd.read_csv("data/processed/nn_train.csv")
        nn_validate = pd.read_csv("data/processed/nn_validate.csv")
        nn_test = pd.read_csv("data/processed/nn_test.csv")
        return nn_train, nn_validate, nn_test
    
    def __load_filefinaldata(self):
        """Load nearest neighbor datasets"""
        train = pd.read_csv("data/processed/final_train.csv")
        validate = pd.read_csv("data/processed/final_validate.csv")
        test = pd.read_csv("data/processed/final_test.csv")
        return train, validate, test
    
    @property
    def loadtrain(self):
        self.__load_loaddata()
        return self._loadtrain
    @property
    def loadfeature(self):
        self.__load_loadfeature()
        return self._loadfeature  
    @property
    def loadfinal(self):
        self.__load_loadfinal()
        return self._loadfinal   
    @property
    def loadvalidate(self):
        self.__load_loaddata()
        return self._loadvalidate
    @property
    def loadtest(self):
        self.__load_loaddata()
        return self._loadtest

    @property
    def nn_train(self):
        self.__load_nn_data()
        return self._nn_train
    @property
    def nn_validate(self):
        self.__load_nn_data()
        return self._nn_validate
    @property
    def nn_test(self):
        self.__load_nn_data()
        return self._nn_test

    @property
    def featuretrain(self):
        self.__load_loadfeature()
        return self._featuretrain
    @property
    def featurevalidate(self):
        self.__load_loadfeature()
        return self._featurevalidate
    @property
    def featuretest(self):
        self.__load_loadfeature()
        return self._featuretest
      
    @property
    def finaltrain(self):
        self.__load_loadfinal()
        return self._finaltrain
    @property
    def finalvalidate(self):
        self.__load_loadfinal()
        return self._finalvalidate
    @property
    def finaltest(self):
        self.__load_loadfinal()
        return self._finaltest

# Global instance
data_manager = DataManager()
