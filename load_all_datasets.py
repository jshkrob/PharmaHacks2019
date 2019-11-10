# standard library dependencies 
from typing import Mapping
import os

# external dependencies
import pandas as pd 

def load_all_datasets(data_dir:str=None, must_have_substring:str="cleaned_",
                    verbose:bool=False) -> Mapping[str, pd.DataFrame]:
    """
    Loads and the contents of every dataset (tab-separated .txt files) as a 
    pandas DataFrame and saves them into a dictionary of (filename:pd.DataFrame)
    (key:value) pairs.
    ----------
    Parameters:
        data_dir:   optional string path to the data directory.
                    defaults to None, which gets replaced by the current 
                    directory.
        must_have_substring:    optional string that all files to load must contain.
                                defaults to "cleaned_"
        verbose:    boolean indicator of verbosity.
                    defaults to False.
    ----------
    Returns:
        dataset_data:   dictionary of (filename: pandas.DataFrame) (key:value) 
                        pairs.
    ----------
    Doctest:
    >>> datasets_dictionary = load_all_datasets(r"data") # indicates that the data folder is a subfolder of the current folder
    >>> datasets_dictionary = load_all_datasets(r"path/to/the/data/folder")
    """
    # if no data_dir argument is given, 
    # we assume we need to use the current directory
    # (the directory which this program is saved in)
    if data_dir is None: 
        data_dir = os.getcwd()

    dataset_data = {}
    for diritem in os.listdir(data_dir):
        # don't look into folders
        if not os.path.isfile(os.path.join(data_dir, diritem)): continue
        
        # don't look into zipped files
        if ".zip" in diritem: continue
        
        # only consider files that have "cleaned_" and ".txt" in their name
        elif must_have_substring in diritem and ".txt" in diritem:

            # load the contents of the file
            df = pd.read_csv(
                os.path.join(data_dir, diritem),
                index_col=0,
                header=0,
                sep='\t'
            )
            if verbose: print(f"\n{diritem}: {df.shape}")
            dataset_data[diritem] = df
    if verbose: print(f"\nfinished compiling {len(dataset_data)} datasets")
    return dataset_data

def compile_datasets(data_dir:str=None, must_have_substring:str="cleaned_",
                    verbose:bool=False) -> Mapping[str,pd.DataFrame]:
    """
    Depreciated alias to the load_all_datasets function.
    Loads and the contents of every dataset (tab-separated .txt files) as a 
    pandas DataFrame and saves them into a dictionary of (filename:pd.DataFrame)
    (key:value) pairs.
    ----------
    Parameters:
        data_dir:   optional string path to the data directory.
                    defaults to None, which gets replaced by the current 
                    directory.
        must_have_substring:    optional string that all files to load must contain.
                                defaults to "cleaned_"
        verbose:    boolean indicator of verbosity.
                    defaults to False.
    ----------
    Returns:
        dataset_data:   dictionary of (filename: pandas.DataFrame) (key:value) 
                        pairs.
    ----------
    Doctest:
    >>> datasets_dictionary = compile_datasets(r"data") # indicates that the data folder is a subfolder of the current folder
    >>> datasets_dictionary = compile_datasets(r"path/to/the/data/folder")
    """
    return load_all_datasets(data_dir=data_dir, must_have_substring=must_have_substring,
                            verbose=verbose)