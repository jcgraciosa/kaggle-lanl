########################################################
# Class definition for random forest model for predicting
# laboratory earthquakes. Entry to LANL competition in Kaggle
# Author: Juan Carlos Graciosa
#
# Revision history:
# Date          Version         Author      Notes
# -----------------------------------------------------
# 04/29/2019    0.1             jcg         - transfer to class
########################################################

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import moment, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class RFQuake(object):

    def __init__(self, samp_freq = 4e6): # is this needed?

        self.samp_freq = samp_freq
        self.samp_period = 1./samp_freq

        return

    def read_raw_data(self, fdir, rd_all = True, st_idx = 0, ed_idx = 10000):

        self.tr_data = pd.read_csv(fdir)

        if not rd_all:
            self.tr_data = self.tr_data[st_idx: ed_idx]

        return

    def read_features(self, fdir):

        self.features = pd.read_csv(fdir)

        return

    def train_model(self, test_size = 0.2, rf_depth = 2, rf_num_est = 100, rand_seed = 234):

        y_all = np.asarray(self.features['y'])
        X_all = np.asmatrix(self.features.drop(['y'], axis = 1))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_all, y_all, test_size = test_size, random_state = rand_seed)
        self.regr = RandomForestRegressor(max_depth = rf_depth, random_state = rand_seed, n_estimators = rf_num_est)
        self.regr.fit(self.X_train, self.y_train)

        print("Done training random regressor model...")
        print("Importance of features: ", self.regr.feature_importance_)

        return

    def test_model(self):

        self.y_pred = self.regr.predict(self.X_test)

        # FIXME: evaluate how well the model predicts the time to time_to_failure

        return

    def compute_features(self, window_size = 1.8, offset = 0.18, wr_all = True, fdir = './features.csv'): # window size and offset are in seconds

        window_elem = int(window_size*self.samp_freq)
        stride_elem = int(offset*self.samp_freq)

        print("elem per window: ", window_elem)
        print("elem per stride: ", stride_elem)

        ttf = np.asarray(self.tr_data['time_to_failure'])
        sig = np.asarray(self.tr_data['acoustic_data'])

        num_samp = ttf.shape[0]
        num_group = num_samp//stride_elem

        indexer = np.arange(window_elem)[None, :] + stride_elem*np.arange(num_group + 1)[:, None]

        # find if dimension has been exceeded (what if it has not been that exceeded)
        # just make sure that it is always exceeded - better to remove afterwards
        a = indexer <= (num_samp - 1)
        b = np.sum(a,axis=1)

        max_idx = np.argmax(b < window_elem) # find index we should start cutting

        indexer = indexer[:max_idx]
        ttf = ttf[indexer]
        sig = sig[indexer]

        print(sig)
        print(ttf)

        # compute statistics here
        features = {}
        print("computing mean ...")
        features['mean'] = np.mean(sig, axis = 1)
        print("computing standard deviation ...")
        features['std'] = np.std(sig, axis = 1)
        print("computing variance ...")
        features['var'] = np.var(sig, axis = 1)
        print("computing 1st moment ...")
        features['mom1'] = moment(sig, axis = 1, moment = 1)
        print("computing 2nd moment ...")
        features['mom2'] = moment(sig, axis = 1, moment = 2)
        print("computing 3rd moment ...")
        features['mom3'] = moment(sig, axis = 1, moment = 3)
        print("computing kurtosis ...")
        features['kurt'] = kurtosis(sig, axis = 1)

        # y value - using the last value per group for the ttf - logical choice
        features['y'] = ttf[:, -1]

        print(features)

        self.features = pd.DataFrame(data = features)

        if wr_all:
            self.features.to_csv(fdir, sep = ',')

        return
