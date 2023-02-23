import pickle
import pandas as pd

def pkl2csv(pkl_path):
    f = open(pkl_path,'rb')
    data = pickle.load(f)
    pd.set_option('display.width',None)
    pd.set_option('display.max_rows',None)
    pd.set_option('display.max_colwidth',None)
    data.to_csv(pkl_path[:-4]+'.csv')
    f.close()

pkl2csv("./process_2018_2022_sp500_data_label.pkl")
pkl2csv("./process_2018_2022_data_label.pkl")