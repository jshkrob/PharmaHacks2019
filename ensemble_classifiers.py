from collections import Counter

import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix

def GB_classifier(X_train:np.ndarray, y_train:np.ndarray, X_val:np.ndarray, y_val:np.ndarray,
                n_estimators:int=1_000, max_depth:int=5, random_state:int=42,
                return_baseline_cm:bool=False):
    """
    """
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return train_and_eval_ensemble_classifier(
        clf,
        X_train, 
        y_train, 
        X_val, 
        y_val,
        return_baseline_cm=return_baseline_cm
    )

def RF_classifier(X_train:np.ndarray, y_train:np.ndarray, X_val:np.ndarray, y_val:np.ndarray,
                n_estimators:int=1_000, max_depth:int=5, random_state:int=42,
                return_baseline_cm:bool=False):
    """
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        random_state=random_state
    )
    clf.fit(X_train, y_train)
    return train_and_eval_ensemble_classifier(
        clf,
        X_train, 
        y_train, 
        X_val, 
        y_val,
        return_baseline_cm=return_baseline_cm
    )

def train_and_eval_ensemble_classifier(clf, X_train:np.ndarray, y_train:np.ndarray, X_val:np.ndarray, y_val:np.ndarray,
                n_estimators:int=1_000, max_depth:int=5, random_state:int=42,
                return_baseline_cm:bool=False):
    """
    """
    
    predictions = clf.predict(X_val)
    conf_matrix = confusion_matrix(
        y_val, predictions
    )

    if return_baseline_cm:
        y_train_dist = Counter(y_train)

        distribution_array = np.zeros((len(y_train_dist,))).astype(float)
        for e,freq in enumerate(y_train_dist):
            distribution_array[e] = freq
        distribution_array /= distribution_array.sum()

        random_predictions = np.random.choice(
            len(y_train_dist),
            size=y_val.shape,
            replace=True,
            p=distribution_array
        )
        baseline_conf_matrix = confusion_matrix(y_val, random_predictions)

        return {
            "Model Confusion Matrix": conf_matrix,
            "Model Accuracy": conf_matrix.trace()/conf_matrix.sum(),
            "Baseline Confusion Matrix": baseline_conf_matrix,
            "Baseline Accuracy": baseline_conf_matrix.trace()/baseline_conf_matrix.sum(),
            "Classifier": clf
        }
    else:
        return {
            "Model Confusion Matrix": conf_matrix,
            "Model Accuracy": conf_matrix.trace()/conf_matrix.sum(),
            "Classifier": clf
        }

def NB_classifier(X_train:np.ndarray, y_train:np.ndarray, X_val:np.ndarray,
                    y_val:np.ndarray, onehotencode:bool=True, return_baseline_cm:bool=True):
    """
    """
    if onehotencode:
        X_train_inds = np.arange(X_train.shape[0])
        X_val_inds = np.arange(0, X_val.shape[0]) + X_train.shape[0]

        Xs = np.vstack((X_train, X_val))

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(Xs)
        one_hot_encoded_X = enc.transform(Xs)

        OHE_Xtrain = one_hot_encoded_X[X_train_inds,:]
        OHE_Xval = one_hot_encoded_X[X_val_inds,:]

        X_train = OHE_Xtrain
        X_val = OHE_Xval
    
    nb_clf = BernoulliNB()
    nb_clf.fit(X_train, y_train)
    return train_and_eval_ensemble_classifier(
        clf,
        X_train, 
        y_train, 
        X_val, 
        y_val,
        return_baseline_cm=return_baseline_cm
    )