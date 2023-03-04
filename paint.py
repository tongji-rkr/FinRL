import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

float_cols=['close','open','high','low','turnover','volume','t5_label']

data = pd.read_csv('../Dataset/process_2018_2022_sp500_data_label.csv', index_col = 0)
np_data = np.array(data[float_cols])
#print(data.head())

# 检查是否有nan
print("是否有确实值"+str(np.isnan(np_data).any()))


# 绘制箱型图
# plt.boxplot(np_data, showfliers=False)
# plt.show()