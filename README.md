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

a = tt.TinyTrans()
#onehot编码
info_df = a.onehot_encoder(info_df, ['order_channel','order_type','travel_way'])

#文档编码
goods_type = ['服装类#1', '日用品类#2', '水果类#3', '食品类#4', '文件类#5', '电子产品#6', '鲜花类#7', '保温类#8', \
                  '易碎品#9', '易融化类#10', '海鲜类#11', '大件品类#12', '蛋糕类#13']
info_df = a.special_encoder(info_df, 'wordseg', goods_type)

label = np.random.randint(0, 2, 99999)
label_vector = pd.DataFrame(label, columns=['label'])

#每个桶左右边界信息
boxinfo = a.box_split(data_df, 'distance', 'ef')
#每个桶详细信息，包括桶编号，是否左开右闭，左右边界，正负样本数量，正样本比例，woe，iv
boxdetail = a.cal_result(boxinfo, data_df['distance'], label_vector, all_p, all_n)
#用woe的值替换原有的特征值
info_df = a.woe_encoder(data_df, 'distance', boxdetail)

