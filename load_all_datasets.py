# standard library dependencies 
import os

# external dependencies
import pandas as pd 

def compile_datasets(data_dir:str=None):
    if data_dir is None: 
        data_dir = os.getcwd()
    dataset_data = {}
    for diritem in os.listdir(data_dir):
        if not os.path.isfile(os.path.join(data_dir, diritem)): continue
        if ".zip" in diritem: continue
        elif "cleaned_" in diritem and ".txt" in diritem:
            df = pd.read_csv(
                os.path.join(data_dir, diritem),
                index_col=0,
                header=0,
                sep='\t'
            )
            print(f"\n{diritem}: {df.shape}")
            dataset_data[diritem] = df
    print(f"\nfinished compiling {len(dataset_data)} datasets")
    return dataset_data