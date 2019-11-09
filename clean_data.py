# standard library dependencies
import os
import re
from typing import List 

# external dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# local dependencies

def clean_data(dataframe:pd.DataFrame, ordinalEncoder, columns_to_encode:List[str]=None, unknown_value=0):
    """
    Cleans the pandas DataFrame provided.
    ----------
    Arguments:
        dataframe: pandas DataFrame containing raw data.
        ordinalEncoder: OrdinalEncoder that has been pre-fitted on the _entire_ dataset.
        columns_to_encode:  list of strings corresponding to the name(s) of 
                            column(s) in `dataframe` which can be encoded by
                            sklearn's OrdinalEncoder.
                            None by default, but gets replaced (see code).
    ----------
    Returns:
        dictionary with the following key:value pairs:
            "cleaned_data": a cleaned pandas DataFrame version of `dataframe`.
            "ordinal_encoder": the fitted sklearn OrdinalEncoder used.
            "encoded_columns":  list of strings corresponding to the name(s) of 
                                column(s) in `dataframe` which were encoded by
                                sklearn's OrdinalEncoder.
    ----------
    Doctest:
    >>> import pandas as pd
    >>> raw_data = pd.read_csv(os.path.join(os.getcwd(), "data", "01-2018.txt"), sep='\t', skiprows=1, header=0, index_col=0)
    >>> cleaned_data_dictionary = clean_data(raw_data)
    >>> cleaned_dataframe = cleaned_data_dictionary['cleaned_data']
    >>> assert set(cleaned_dataframe.columns.tolist()) == set(raw_data.columns.tolist())
    """
    if columns_to_encode is None:
        cleaned_column_names = [
            "Age Range",
            "Day Enrollment Received", 
            "Day Enrollment Completed",
            "State Change Day",
            "On Drug Start Day",
            "Last On Drug Day",
            "Re-Engagement Day",
            "Re-Engagement On Drug Start Day"
        ]
        columns_to_encode = list( set(dataframe.columns.tolist()) - set(cleaned_column_names) )
    
    # using scikit-learn's ordinalencoder for features which didn't have extra text
    oed_data = ordinalEncoder.transform(dataframe.loc[:,columns_to_encode].values)
    cleaned_df = pd.DataFrame(oed_data, columns=columns_to_encode, index=dataframe.index)
        
    # manually cleaning leftover columns

    # convert age ranges into integers (e.g. 50-59 -> 5, 60-69 -> 6)
    cleaned_df["Age Range"] = [ int(age_range.split("-")[0]) for age_range in dataframe["Age Range"] ]
    
    # convert things like "Day_42198" into integers by keeping only numbers (e.g. "Day_42198" -> 42198)
    # "Unknown/Never Completed" gets replaced by 0
    cleaned_df["Day Enrollment Received"] = [ unknown_value if "Unknown" in day else int(re.sub("[^0-9]", "", day)) for day in dataframe["Day Enrollment Received"] ]
    cleaned_df["Day Enrollment Completed"] = [ unknown_value if "Unknown" in day else int(re.sub("[^0-9]", "", day)) for day in dataframe["Day Enrollment Completed"] ]
    cleaned_df["State Change Day"] = [ unknown_value if "Unknown" in day else int(re.sub("[^0-9]", "", day)) for day in dataframe["State Change Day"] ]
    cleaned_df["On Drug Start Day"] = [ unknown_value if "Unknown" in day else int(re.sub("[^0-9]", "", day)) for day in dataframe["On Drug Start Day"] ]
    cleaned_df["Last On Drug Day"] = [ unknown_value if "Unknown" in day else int(re.sub("[^0-9]", "", day)) for day in dataframe["Last On Drug Day"] ]
    cleaned_df["Re-Engagement Day"] = [ unknown_value if "Unknown" in day else int(re.sub("[^0-9]", "", day)) for day in dataframe["Re-Engagement Day"] ]
    cleaned_df["Re-Engagement On Drug Start Day"] = [ unknown_value if "Unknown" in day else int(re.sub("[^0-9]", "", day)) for day in dataframe["Re-Engagement On Drug Start Day"] ]
    
    return cleaned_df

def fit_Ordinal_encoder(combined_data_filepath:str=None, columns_to_encode:List[str]=None):
    """
    Fits a scikit-learn.preprocessing OrdinalEncoder on a subset of features (columns)
    on the _entire_ dataset.
    ----------
    Parameters:
        combined_data_filepath: string path to the combined dataset.
                                None by default, but gets replaced (see code).
        columns_to_encode:  list of strings corresponding to the name(s) of 
                            column(s) in `dataframe` which can be encoded by
                            sklearn's OrdinalEncoder.
                            None by default, but gets replaced (see code).
    ----------
    Returns:
        a scikit-learn.preprocessing OrdinalEncoder that has been fit on the specified
        features of the entire dataset.
    ----------
    References:
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder
    ----------
    Doctest:
    >>> ord_enc = fit_Ordinal_encoder()
    """
    if combined_data_filepath is None:
        combined_data_filepath = os.path.join(os.getcwd(), "data", "combined_data.txt")

    all_raw_data = pd.read_csv(combined_data_filepath, sep='\t', skiprows=1, header=0, index_col=0)

    if columns_to_encode is None:
        cleaned_column_names = [
            "Age Range",
            "Day Enrollment Received", 
            "Day Enrollment Completed",
            "State Change Day",
            "On Drug Start Day",
            "Last On Drug Day",
            "Re-Engagement Day",
            "Re-Engagement On Drug Start Day"
        ]
        columns_to_encode = list( set(all_raw_data.columns.tolist()) - set(cleaned_column_names) )
    
    # using scikit-learn's ordinalencoder for features which didn't have extra text
    data_to_encode = all_raw_data.loc[:,columns_to_encode]
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit(data_to_encode.values)
    print("ordinal encoder has been fit")
    return ordinal_encoder

def main():
    """
    """
    data_directory = os.path.join(os.getcwd(), "data")
    ordinal_encoder = fit_Ordinal_encoder()

    files_to_clean = [
        "01-2018.txt", "02-2019.txt", "05-2018.txt", 
        "08-2018.txt", "11-2018.txt", "2019-04.txt", 
        "2019-07.txt", "01-2019.txt", "03-2018.txt", 
        "06-2018.txt", "09-2018.txt", "12-2018.txt", 
        "2019-05.txt", "2019-08.txt", "02-2018.txt", 
        "04-2018.txt", "07-2018.txt", "10-2018.txt", 
        "2019-03.txt", "2019-06.txt", "2019-09.txt"
    ]

    for file in files_to_clean:
        filepath = os.path.join(data_directory, file)
        raw_data_dataframe = pd.read_csv(
            filepath, 
            sep='\t', 
            skiprows=1, 
            header=0, 
            index_col=0
        )

        clean_data_dataframe = clean_data(
            raw_data_dataframe,
            ordinal_encoder
        )

        cleaned_filepath = os.path.join(data_directory, f"cleaned_{file}")
        clean_data_dataframe.to_csv(
            cleaned_filepath,
            sep='\t',
            header=True,
            index=True
        )

        print(f"cleaned version of:\n{filepath}\nhas been saved under:\n{cleaned_filepath}\n")

if __name__ == '__main__':
    main()