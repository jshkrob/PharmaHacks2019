# standard library dependencies
from typing import List, Tuple, Mapping
from tqdm import tqdm 

# external dependencies
import numpy as np
import pandas as pd

# local dependencies 
from load_all_datasets import compile_datasets

def count_changes_in_feature(month_to_dataframe_dict:Mapping[str, pd.DataFrame], months_to_consider:List[str] ):
    """
    """
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

    new_column_names = [ f"delta_{colname}" for colname in deltas_df.columns ]
    deltas_df.columns = new_column_names
    return deltas_df

def get_deltas_df(months_to_consider:List[str], data_folder_path:str="data") -> pd.DataFrame:
    """
    """
    dict_of_dfs = compile_datasets(data_folder_path, must_have_substring="dummy")
    return count_changes_in_feature(dict_of_dfs, months_to_consider)

if __name__ == "__main__":
    months_to_test = [
        "dummytest1.txt",
        "dummytest2.txt",
        "dummytest3.txt"
    ]
    res = get_deltas_df(months_to_test, data_folder_path=r"data/test")
    breakpoint()