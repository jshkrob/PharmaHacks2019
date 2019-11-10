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