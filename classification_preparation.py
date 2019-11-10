# standard library dependencies
from collections import Counter
from typing import Mapping, List

# external dependencies
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# local dependencies
from load_all_datasets import compile_datasets
from compute_deltas_df import get_deltas_df
from featureengineering import add_engineered_features
from find_best_features import find_best_features

def load_dataset_with_engineered_features(path_to_data_folder:str,
                                          must_have_substring:str="cleaned_",
                                          verbose:bool=False) -> Mapping[str,pd.DataFrame]:
    """
    """
    name_to_df_dict = compile_datasets(path_to_data_folder, 
        must_have_substring=must_have_substring,
        verbose=verbose
    )
    if verbose: print(f"loaded {len(name_to_df_dict)} datasets; adding engineered features")

    for dataset_name, dataset_dataframe in name_to_df_dict.items():
        name_to_df_dict[dataset_name] = add_engineered_features(dataset_dataframe)

    if verbose: print("added engineered features to all datasets")
    return name_to_df_dict

def return_data_to_predict_at_month(name_to_df_dict:Mapping[str,pd.DataFrame],
                                    prediction_month:str,
                                    starting_month:str="cleaned_01-2018.txt",
                                    specific_months:List[str]=None,
                                    verbose:bool=False):
    """
    """
    assert prediction_month in name_to_df_dict.keys()
    chrono_ordered_datasets = [
        "cleaned_01-2018.txt",
        "cleaned_02-2018.txt",
        "cleaned_03-2018.txt",
        "cleaned_04-2018.txt",
        "cleaned_05-2018.txt",
        "cleaned_06-2018.txt",
        "cleaned_07-2018.txt",
        "cleaned_08-2018.txt",
        "cleaned_09-2018.txt",
        "cleaned_10-2018.txt",
        "cleaned_11-2018.txt",
        "cleaned_12-2018.txt",
        "cleaned_01-2019.txt",
        "cleaned_02-2019.txt",
        "cleaned_2019-03.txt",
        "cleaned_2019-04.txt",
        "cleaned_2019-05.txt",
        "cleaned_2019-06.txt",
        "cleaned_2019-07.txt",
        "cleaned_2019-08.txt"
    ]
    index_of_starting_month = chrono_ordered_datasets.index(starting_month)
    index_of_prediction_month = chrono_ordered_datasets.index(prediction_month)

    months_to_consider_in_delta = chrono_ordered_datasets[index_of_starting_month:index_of_prediction_month]
    delta_df = get_deltas_df(
        months_to_consider_in_delta,
        name_to_df_dict
    )

    all_useable_patients = delta_df.index.tolist()
    if verbose: print(f"{len(all_useable_patients)} useable patients")

    # keeping only the patients for which we have delta-type data
    data_at_month_to_predict = name_to_df_dict[prediction_month].loc[all_useable_patients]

    all_data = pd.concat([ delta_df, data_at_month_to_predict ], axis=1)
    return all_data

def prepare_to_predict_feature(feature_name:str,
                                all_useable_data:pd.DataFrame,
                                min_frequency:int=3,
                                verbose:bool=False):
    """
    """
    assert feature_name in all_useable_data.keys()
    y = all_useable_data.loc[:,feature_name]

    label_distribution = Counter(y.values)
    label_to_frequency_tuples = label_distribution.most_common()

    for (label, frequency) in label_to_frequency_tuples[::-1]:
        if frequency < min_frequency:
            if verbose: 
                explanation = (
                    f"class {label} has too few instances to be split into training/validation/testing sets"
                    f"\nreplacing {label} labels with {label_to_frequency_tuples[0][0]} (the most common label)"
                )
            print(explanation)

            y.iloc[ np.where( y == label ) ] = label_to_frequency_tuples[0][0]

    features_df = all_useable_data.drop(columns=[feature_name])
    assert feature_name not in features_df.columns

    return features_df, y

def split_into_training_validating_testing( X:pd.DataFrame,
                                            y:pd.DataFrame,
                                            train_fraction:float=0.85,
                                            test_fraction:float=0.15,
                                            validation_fraction:float=0.2,
                                            use_strat_splits:bool=False,
                                            nsplits:int=10,
                                            random_state:int=42,
                                            verbose:bool=False):
    """
    """
    assert train_fraction + test_fraction == 1.0

    _, X_test_indices, _, y_test_indices = train_test_split(
        np.arange(X.shape[0]),
        np.arange(y.shape[0]),
        train_size=train_fraction,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y.values
    )

    X_test_data = X.iloc[X_test_indices,:]
    y_test_data = y.iloc[y_test_indices]

    X_train_val_data = X.drop(X.index[X_test_indices])
    y_train_val_data = y.drop(y.index[y_test_indices])

    if use_strat_splits:
        sss = StratifiedShuffleSplit(
            n_splits=nsplits, 
            test_size=(train_fraction - (train_fraction*validation_fraction)), 
            random_state=random_state
        )
        for train_index, validation_index in sss.split(X_train_val_data, y_train_val_data):
            X_train, X_val = X[train_index], X[validation_index]
            y_train, y_val = y[train_index], y[validation_index]
            # print(Counter(y_train), '\n', Counter(y_test))
            yield {
                "X train": X_train,
                "X validate": X_val,
                "X test": X_test_data,
                "y train": y_train,
                "y validate": y_val,
                "y test": y_test_data
            }
    else:
        X_train_indices, X_validation_indices, y_train_indices, y_validation_indices = train_test_split(
            np.arange(X_train_val_data.shape[0]),
            np.arange(y_train_val_data.shape[0]),
            test_size=0.2,
            random_state=42,
            stratify=y_train_val_data
        )

        X_train = X_train_val_data.iloc[X_train_indices,:]
        X_val = X_train_val_data.iloc[X_validation_indices,:]
        y_train = y_train_val_data.iloc[y_train_indices]
        y_val = y_train_val_data.iloc[y_validation_indices]

        if verbose:
            print(f"dimensions of testing dataset: {X_test_data.shape}, {y_test_data.shape}")
            print(f"dimensions of validating dataset: {X_val.shape}, {y_val.shape}")
            print(f"dimensions of training dataset: {X_train.shape}, {y_train.shape}")
            print(f"dimensions of original dataset: {X.shape}")

        assert X_train.shape[0] + X_val.shape[0] + X_test_data.shape[0] == X.shape[0]
        assert y_train.shape[0] + y_val.shape[0] + y_test_data.shape[0] == y.shape[0]

        return {
            "X train": X_train,
            "X validate": X_val,
            "X test": X_test_data,
            "y train": y_train,
            "y validate": y_val,
            "y test": y_test_data
        }

