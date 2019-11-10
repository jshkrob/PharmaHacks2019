# standard library dependencies
from typing import List, Tuple, Mapping
from tqdm import tqdm 

# external dependencies
import numpy as np
import pandas as pd

# local dependencies 
from load_all_datasets import compile_datasets

def count_changes_in_feature(month_to_dataframe_dict:Mapping[str, pd.DataFrame], 
                            months_to_consider:List[str]) -> pd.DataFrame:
    """
    Creates the deltas pandas DataFrame for the datasets in `month_to_dataframe_dict`
    that are specified by the strings (keys) in `months_to_consider`
    ----------
    Parameters:
        month_to_dataframe_dict:    dictionary of (filename:pd.DataFrame) (key:value)
                                    pairs.
        months_to_consider: list of strings corresponding to the keys of dataframes in 
                            `month_to_dataframe_dict` which need to be used in the delta
                            calculation.
    ----------
    Returns:
        deltas_df:  pd.DataFrame whose rows correspond to patients, columns to features, 
                    and where the deltas_df[p,f] indicates how many times patient p's
                    fth feature changed over the months specified in `months_to_consider`.
    """
    for month in months_to_consider:
        try:
            assert month in month_to_dataframe_dict.keys()
        except AssertionError as ae:
            raise KeyError(f"Key {month} not found in the dictionary's keys:\n{','.join(month_to_dataframe_dict.keys())}") from ae

    deltas_df = pd.DataFrame(
        np.zeros(
            (
                month_to_dataframe_dict[months_to_consider[0]].shape[0], 
                month_to_dataframe_dict[months_to_consider[0]].shape[1] + 1)
        ), # +1 is for the "Number of Months"
        index=month_to_dataframe_dict[months_to_consider[0]].index,
        columns=month_to_dataframe_dict[months_to_consider[0]].columns.tolist() + ["Number of Months"]
    )

    deltas_df.iloc[:,-1] = 1

    last_month_df = month_to_dataframe_dict[months_to_consider[0]]

    for month in tqdm(months_to_consider[1:]):
        dataframe_for_this_month = month_to_dataframe_dict[month]
        
        # find patient to update
        common_patient_ids = [
            patient_id for patient_id in last_month_df.index
            if patient_id in dataframe_for_this_month.index
        ]
        
         # set(dataframe_for_this_month.index.tolist()).intersection(set(last_month_df.index.tolist()))

        subdf_for_this_month_to_compare = dataframe_for_this_month.loc[common_patient_ids]
        comparison_df = subdf_for_this_month_to_compare != last_month_df.loc[common_patient_ids]
        comparison_df["Number of Months"] = 0
        deltas_df += comparison_df.astype(int)

        # find patients which need to be added
        new_patients = [
            patient_id for patient_id in dataframe_for_this_month.index
            if patient_id not in last_month_df.index
        ]
        
        # create a "null" dataframe to append to deltas_df in one single append command
        new_patients_df = pd.DataFrame(
            np.zeros((len(new_patients), deltas_df.shape[1])),
            index=new_patients,
            columns=deltas_df.columns
        )
        
        # merge this new "no deltas" dataframe to the deltas_df dataframe
        deltas_df = deltas_df.append(new_patients_df)

        # increment the "Number of Months" for every patient seen so far
        deltas_df.iloc[:,-1] += 1

        # update last_month_df to be this month's df 
        last_month_df = dataframe_for_this_month

    # update the column names to include the 'delta_' prefix
    new_column_names = [ f"delta_{colname}" for colname in deltas_df.columns ]
    deltas_df.columns = new_column_names
    return deltas_df

def get_deltas_df(months_to_consider:List[str], dictionary_of_dfs:Mapping[str, pd.DataFrame]=None, 
                data_folder_path:str=None, must_have_substring:str="cleaned_") -> pd.DataFrame:
    """
    Wrapper around the `count_changes_in_feature` function.
    ----------
    Parameters:
        months_to_consider: list of strings corresponding to the keys of dataframes in 
                            `month_to_dataframe_dict` which need to be used in the delta
                            calculation.
        dictionary_of_dfs:  optional dictionary of (filename:pd.DataFrame) 
                            (key:value) pairs.
        data_folder_path:   optional string path to the data directory.
                            defaults to None, which gets replaced by the current 
                            directory.
        must_have_substring: optional string that all files to load must contain.
                                defaults to "cleaned_".
    ----------
    Returns:
        pd.DataFrame whose rows correspond to patients, columns to features, 
        and where the deltas_df[p,f] indicates how many times patient p's
        fth feature changed over the months specified in `months_to_consider`.
    """
    if dictionary_of_dfs:
        assert isinstance(dictionary_of_dfs, dict) 
    elif data_folder_path is not None:
        print("Loading datasets without engineered features...")
        dictionary_of_dfs = compile_datasets(data_folder_path, must_have_substring=must_have_substring)
    
    return count_changes_in_feature(dictionary_of_dfs, months_to_consider)

if __name__ == "__main__":
    months_to_test = [
        "cleaned_01-2018.txt",
        "cleaned_02-2018.txt",
        "cleaned_03-2018.txt"
    ]
    res = get_deltas_df(months_to_test, data_folder_path=r"data")
