# external dependencies
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def find_best_features(X:np.ndarray, y:np.ndarray) -> np.ndarray:
    '''
    Parameters:
        X: numpy array 
        y: numpy array of response variable
    Returns:
        The numpy array returned by sklearn.feature_selection.SelectFromModel
    '''
    # Selector obj using Random Forest classifier
    rf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0)
    sfm = SelectFromModel(rf, threshold=0.15)
    
    # Train Selector obj
    sfm.fit(X, y)
    
    # Output Data Subset w/ Important Features
    return sfm.transform(X)