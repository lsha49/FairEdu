
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import math
import imblearn
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from deslib.util.instance_hardness import kdn_score
from numpy import where
import random
from math import log2, log, sqrt
from abroca import *
from scipy.spatial import distance
from statistics import stdev, mean

class test_cbt(object):
    def preprocessing(self): 
        # @todo import your data file
        filename = ''; 
        # useLabel = ''; 
        # Corpus = pd.read_csv(filename, encoding='') 

        # @todo add your test size and random counter                                                                 
        # self.Train_X, self.Test_X, self.Train_Y, self.Test_Y = model_selection.train_test_split(Corpus.to_numpy(), Corpus[useLabel], test_size=, random_state=)        

        # @todo generate samples by Class Balancing Techniques (CBTs)
        # self.cbt()
        
        # @todo calculate KDN and hardness bias (hard-bias)
        # self.calKDN()
                

    def cbt(self, X, Y, G):            
        balanceMode = ''
            
        if balanceMode == 'C': # class oversample
            X, Y = SMOTE().fit_resample(X, Y)
            
        if balanceMode == 'CU': # class undersample
            X, Y = NearMiss().fit_resample(X, Y) 
            
        if balanceMode == 'OV': # over sample strategy
            GY = G.astype(str) + Y.astype(str)
            X, GY = SMOTE().fit_resample(X, GY)  
            G = GY.str[0].astype(int); Y = GY.str[1].astype(int)

        if balanceMode == 'UN': # under sample strategy
            GY = G.astype(str) + Y.astype(str)
            X, GY = NearMiss().fit_resample(X, GY)  
            G = self.GY.str[0].astype(int); Y = self.GY.str[1].astype(int)



    def calKDN(self, features, labels,  List00Index, List01Index, List10Index, List11Index): 
        kdnResult = kdn_score(features, labels, 5)
        kl_pq0 = distance.jensenshannon(kdnResult[0][List00Index], kdnResult[0][List01Index])
        kl_pq1 = distance.jensenshannon(kdnResult[0][List10Index], kdnResult[0][List11Index])
        return (kl_pq0 + kl_pq1)/2


    # ABROCA related calculations
    def getDfToComputeAbroca(self, predicted, predictionProb):
        predictionDataframe = pd.DataFrame(predicted, columns = ['predicted'])
        predictionDataframe['prob_1'] = pd.DataFrame(predictionProb)[1]
        predictionDataframe['label'] = self.Test_Y.tolist()
        predictionDataframe['gender'] = self.Test_G.astype(str)
        return predictionDataframe

    # ABROCA calculation
    def computeAbroca(self, abrocaDf):
        slice = compute_abroca(abrocaDf, 
                            pred_col = 'prob_1' , 
                            label_col = 'label', 
                            protected_attr_col = 'gender',
                            majority_protected_attr_val = '2',
                            compare_type = 'binary', # binary, overall, etc...
                            n_grid = 10000,
                            plot_slices = False)
        print(slice)
