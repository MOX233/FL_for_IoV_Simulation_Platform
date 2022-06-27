
import pandas as pd
import os
import pickle

df = pd.read_pickle("./data/features/forecasting_features_val.pkl")
save_path = "./ground_truth_data"
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
val_gt = {}
for i in range(len(df)):
    seq_id = df.iloc[i]['SEQUENCE']
    curr_arr = df.iloc[i]['FEATURES'][20:][:, 3:5]
    val_gt[seq_id] = curr_arr

with open(save_path + '/ground_truth_val.pkl', 'wb') as f:
    pickle.dump(val_gt, f)