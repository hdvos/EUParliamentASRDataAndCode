import os
import pandas as pd
import numpy as np
root_folder = '/home/hugo/MEGA/work/ASR/EUParliamentASR/hotwords_experiment/processed_data'

def get_hotwords_weight(filename:str) -> int:
    a = os.path.splitext(filename)[0]
    a = a.split('_')[1]
    a = int(a)
    return a

def get_motherfile_id(filename):
    basename = os.path.basename(filename)
    barename = os.path.splitext(basename)[0]
    motherfile_id = barename.split('_')[-1]

    return motherfile_id



for root, folders,files in os.walk(root_folder):

    # print(files)
    files = sorted(files, key=lambda x:get_hotwords_weight(x))
    print(files)

    for i, file in enumerate(files):
        fullpath = os.path.join(root, file)
        df = pd.read_csv(fullpath, sep='\t', index_col=0)
        # print(df.columns)
        pivot_table = pd.pivot_table(df,values= ['recall', 'unrecognized_words', 'false_positive_hotwords', 'precision','WER'], index='identifier', aggfunc=np.mean)
        print('-------------------------------------------------')
        print(df[df.identifier == 20140701])
        print(pivot_table)
        print('-------------------------------------------------')
        input()