import pandas as pd
import numpy as np
MAE_sum = 0
RMSE_sum = 0
num_count = 0
path_est = r"/home/bestlab/Desktop/shoot_moon/PR_RR_csv/PR_jihong_paper.csv"#PR_shints_not_lib_pca_1.csv
# path_est = r"/home/bestlab/Desktop/final_com_v1/data_0608/shints_PR.csv"
path_gt = r"/media/bestlab/Transcend/Datasets/Dark/0105_jihong/Front/Polar_HeartRate/2023-01-05_15-38-46.CSV"
df_estimate = pd.read_csv(path_est, header=None)
# df_estimate = df_estimate.iloc[10:,]
df_estimate = df_estimate.to_numpy()
print(df_estimate)
df_gt = pd.read_csv(path_gt, header=None)
df_gt = df_gt.iloc[3:,2]
df_gt = df_gt.astype(int)
df_gt = df_gt.to_numpy()
print(df_gt)
for i in range(15,len(df_estimate)):
    MAE_sum += abs(df_estimate[i-15][0]-df_gt[i])
    RMSE_sum += pow(df_estimate[i-15][0]-df_gt[i],2)
    num_count += 1
print('MAE',round(MAE_sum/num_count,2))
print('RMSE',round(np.sqrt(RMSE_sum/num_count),2))