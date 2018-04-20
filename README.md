#简介

# 安装
   
    $ pip install --upgrade pip
    $ pip install git+https://github.com/yuqingsheng/tinytrans

# Demo

~~~python
import tinytrans as tt
import pandas as pd

info_df = pd.read_csv("order_info.csv", header=0)
info_df = info_df.iloc[:, [0, 2, 3, 4, 5, 6, 8, 9, 12]]
info_df.columns = ['order_id', 'order_channel', 'order_type', 'distance', 'weight', 'travel_way', 'lat', 'lng', 'wordseg']

a = tt.TinyTrans(info_df)
b = a.onehot_encoder(['order_channel','order_type','travel_way'])
