# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from load_all_datasets import compile_datasets

def correct_shift(datasets):
    a = datasets["Last On Drug Day"] 
    b = datasets["On Drug Start Day"]
    var = a-b if a != 0 else b
    return var

def add_features(list_of_dfs):
    for index, df in enumerate(list_of_dfs):
        df["Enrollment Time"] = df["Day Enrollment Completed"] - df["Day Enrollment Received"]
        df[ df["Enrollment Time"] < 0 ] = 0
        df["On Drug Time"] = df[["Last On Drug Day", "On Drug Start Day"]].apply(correct_shift, axis=1)
        df["# of Payment Methods"] = max(0, df[["Payment Method #4", "Payment Method #5", "Payment Method #1", "Payment Method #3", "Payment Method #2"]].sum(axis=1) )
        list_of_dfs[index] = df
        
    return list_of_dfs

def dosagefrequencyindex(list_of_dfs):
    for index, df in enumerate(list_of_dfs):
        df["DosageFrequencyIndex"] = 3*df["Dosage"] + df["Frequency"]
        list_of_dfs[index] = df
        
    return list_of_dfs

def testingdosagefrequencyindex():
    dic = compile_datasets("data")
    df = dic["cleaned_09-2018.txt"]
    testdflist = dosagefrequencyindex([df])
    print(testdflist[0][["Dosage", "Frequency", "DosageFrequencyIndex"]].head())

def add_engineered_features(df):
    with_new_features = df.copy(deep=True)
    with_new_features["Enrollment Time"] = with_new_features["Day Enrollment Completed"] - with_new_features["Day Enrollment Received"]
    with_new_features["On Drug Time"] = with_new_features[["Last On Drug Day", "On Drug Start Day"]].apply(correct_shift, axis=1)
    with_new_features["# of Payment Methods"] = with_new_features[["Payment Method #4", "Payment Method #5", "Payment Method #1", "Payment Method #3", "Payment Method #2"]].sum(axis=1)
    with_new_features["DosageFrequencyIndex"] = 3*with_new_features["Dosage"] + with_new_features["Frequency"]

    return with_new_features

if __name__ == "__main__":
    testingdosagefrequencyindex()

