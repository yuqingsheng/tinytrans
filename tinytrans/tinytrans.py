#coding=utf8
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from collections import defaultdict
import math
import copy

class TinyTrans(object):
    def __init__(self):
        pass

    # 文档编码
    # input:
    #       data:原始数据dataframe
    #       colname:需要转为文档编码的特征列
    #       typelist:该列特征的所有取值列表
    #       sep:待转换特征列里各个取值之间的分隔符
    # output:
    #       重新编码之后的多个特征列，dataframe格式


    def special_encoder(self, data, colname, typelist, sep='|'):
        length = len(typelist)
        arr = np.array([0] * length)
        total_arr = []
        total_arr.append(arr)
        for i in range(length):
            arr = np.array([0] * length)
            arr[i] = 1
            total_arr.append(arr)

        columns = [colname + str(i) for i in range(1, length + 1)]
        type_dict = dict(zip(typelist, list(range(1, length + 1))))

        total_result = []
        for x in data[colname].values:
            type_code = np.array([0] * length)
            if x and isinstance(x, str):
                arr = x.split(sep)
                for type_tmp in arr:
                    type_code += total_arr[type_dict[type_tmp]]
            total_result.append(type_code)
        new_df = DataFrame(total_result, columns=columns)
        data = pd.concat([data, new_df], axis=1)
        return data


    # 独热编码
    # input:
    #       data:原始数据dataframe
    #       collist:需要转为onehot编码的特征列list
    # output:
    #       转换完之后的data
    def onehot_encoder(self, data, collist):
        for col in collist:
            data[col] = data[col].astype('str')
        return pd.get_dummies(data, dummy_na=True, columns=collist)

    # woe编码
    def woe_encoder(self, data, colname, boxdetail):
        origin_feature_vector = data[colname].values
        feature_vector = copy.deepcopy(origin_feature_vector)
        for i in range(len(boxdetail)):
            bin_box = boxdetail[i]
            left_edges = float(bin_box['left_edges'])
            right_edges = float(bin_box['right_edges'])
            woe = bin_box['woe']
            idx_1 = np.where(origin_feature_vector > left_edges)[0]
            tmp_f = origin_feature_vector[idx_1]
            idx_2 = np.where(tmp_f <= right_edges)[0]
            idx = idx_1[idx_2]
            feature_vector[idx] = float(woe)
        data[colname] = feature_vector
        return data

    # 分桶归一化，目的是为了解决长尾问题，常用的归一化方式解决不了长尾问题，使得最后特征值集中在一块，降低特征的区分能力
    # 对于特征值分布不均匀的数据，需要先做分桶，然后做分桶归一化，例如:
    # 总共分了n个桶，而特征xi属于其中的第bi(bi ∈ {0, ..., n - 1})个桶，则特征xi最终会归一化成 bi/n
    # 归一化操作和woe编码操作二选一
    def normalization_boxdata(self, data, colname, boxinfo):
        origin_feature_vector = data[colname].values
        feature_vector = copy.deepcopy(origin_feature_vector)
        for i in range(len(boxinfo)):
            bin_box = boxinfo[i]
            left_edges = float(bin_box['left_edges'])
            right_edges = float(bin_box['right_edges'])
            value = float(i) / len(boxinfo)
            idx_1 = np.where(origin_feature_vector > left_edges)[0]
            tmp_f = origin_feature_vector[idx_1]
            idx_2 = np.where(tmp_f <= right_edges)[0]
            idx = idx_1[idx_2]
            feature_vector[idx] = float(value)
        data[colname] = feature_vector
        return data

    # 获取分桶的详细信息
    def transform_to_boxdata(self, boxinfo):
        box_data = []
        left_edge = float("-inf")
        right_edge = float("inf")

        try:
            if isinstance(boxinfo, dict):
                keylist = boxinfo.keys()
                keynum = len(keylist)
                if keynum == 1:
                    box_data.append([left_edge, right_edge])
                    return box_data
                keylist = sorted(keylist)
                i = 1
                for key in keylist:
                    if i == 1:
                        right_edge = max(boxinfo[key])
                    elif i < keynum:
                        left_edge = right_edge
                        right_edge = max(boxinfo[key])
                    else:
                        left_edge = right_edge
                        right_edge = float("inf")
                    i += 1
                    box_data.append([left_edge, right_edge])
            elif isinstance(boxinfo, list):
                if len(boxinfo) == 1:
                    box_data.append([left_edge, right_edge])
                    return box_data
                boxinfo.sort()
                for value in boxinfo:
                    box_data.append([left_edge, value])
                    left_edge = value
                box_data.append([boxinfo[-1], right_edge])
        except:
            box_data = []
        return box_data

    #等频分桶
    def box_split_ef(self, data, colname, box_num):
        init_map = defaultdict(list)
        feature_list = data[colname]
        sample_nums = len(feature_list)
        value_counts = {}
        # 特征值排序
        for sample_num in range(sample_nums):
            cur_value = feature_list[sample_num]
            if cur_value not in value_counts:
                value_counts[cur_value] = 0
            value_counts[cur_value] += 1
        sorted_value_counts = sorted(value_counts.items(), key=lambda d: d[0])

        bin_length = sample_nums * 1.0 / box_num  # 每个桶里的数量
        cur_bin_num = 0
        cur_samples = 0
        ##在做桶合并时需要考虑剩余桶及样本量，特别是最后一个桶，有可能样本数极少
        for value, count in sorted_value_counts:
            if count >= bin_length:
                cur_bin_num += 1
                init_map[cur_bin_num] = [value]
                cur_bin_num += 1
                cur_samples = 0
            else:
                cur_samples += count
                init_map[cur_bin_num].append(value)
                if cur_samples >= bin_length:
                    cur_bin_num += 1
                    cur_samples = 0
        return init_map

    #等宽分桶
    def box_split_ew(self, data, colname, box_num):
        feature_list = data[colname]
        feature_set = np.unique(feature_list)
        feature_set.sort()
        lens = len(feature_set)
        step = int(lens / box_num)
        if lens < box_num:
            box_num = lens
            step = 1

        init_map = {}
        for i in range(0, box_num):
            init_map[i] = []
            for j in range(0, step):
                init_map[i].append(feature_set[i * step + j])
        # 最后一些多余分桶处理
        lasts = lens % box_num
        if lasts > step:
            normal_tong = int(lasts / step)
            for i in range(0, normal_tong):
                init_map[box_num] = []
                for j in range(0, step):
                    init_map[box_num].append(feature_set[box_num * step + j])
                box_num += 1
        lasts = lens % box_num
        for i in range(0, lasts):
            init_map[box_num - 1].append(feature_set[-1 - i])
        return init_map

    #分桶
    #默认等频分桶
    def box_split(self, data, colname, method='ef', box_num=10):
        if method == 'ew':
            return self.transform_to_boxdata(self.box_split_ew(data, colname, box_num))
        else:
            return self.transform_to_boxdata(self.box_split_ef(data, colname, box_num))

    #计算woe和iv值
    def cal_woe_iv(self, good_counts, bad_counts, all_good, all_bad):
        try:
            py1 = bad_counts * 1.0 / all_bad
            py2 = good_counts * 1.0 / all_good
            woe = math.log(py1 / py2)
            iv = (py1 - py2) * woe
        except Exception as e:
            woe = 0
            iv = 0
        return woe, iv

    def cal_result(self, bin_result, feature_list, label_list, all_p, all_n,is_left_open=1, p_is_good=1):
        feature_box = []
        for idx in range(len(bin_result)):
            box_map = {}
            x = bin_result[idx]
            if is_left_open:  # 左开右闭
                f_idx1 = np.where(feature_list > x[0])[0]
                tmp_f = feature_list[f_idx1]
                f_idx2 = np.where(tmp_f <= x[1])[0]
            else:
                f_idx1 = np.where(feature_list >= x[0])[0]
                tmp_f = feature_list[f_idx1]
                f_idx2 = np.where(tmp_f < x[1])[0]

            f_idx = f_idx1[f_idx2]

            cur_label = label_list.ix[f_idx]
            p = len(np.where(cur_label > 0)[0])
            n = len(np.where(cur_label < 1)[0])
            if p_is_good == 1:
                if p == 0:
                    return {}
                woe, iv = cal_woe_iv(p, n, all_p, all_n)
            else:
                if n == 0:
                    return {}
                woe, iv = cal_woe_iv(n, p, all_n, all_p)

            total = p + n
            prate = p / float(total)
            box_map['bin_num'] = idx
            box_map['left_edges'] = x[0]
            box_map['right_edges'] = x[1]
            box_map['p'] = p
            box_map['n'] = n
            box_map['total'] = total
            box_map['prate'] = prate
            box_map['woe'] = woe
            box_map['iv'] = iv
            box_map['is_left_open'] = is_left_open
            box_map['all_p'] = all_p
            box_map['all_n'] = all_n

            feature_box.append(box_map)
        return feature_box

    #异常值处理
    #拉依达准则
    #input:
    #   max_outlier_rate:异常值占总记录的比例阈值，低于这个阈值的才会被过滤
    def outlier_filter(self, data, colname, max_outlier_rate=0.05):
        miu   = data[colname].mean()
        lmbda = data[colname].std()
        left_out  = miu - 3*lmbda
        right_out = miu + 3*lmbda
        tmp_df = data[data[colname] >= left_out]
        tmp_df = tmp_df[tmp_df[colname] <= right_out]

        # 异常占比低于指定的比例，才进行删除
        if (1 - 1.0 * len(tmp_df) / len(data)) <= max_outlier_rate:
            return tmp_df
        return data

    #按照格式生成最后的数据
    #需要dataframe中必须有'label'列
    #格式如下：
    #label 0:0.1234 1:0.234 2:0.453 ...... n:0.9876
    def final_feature_file(self, data, line_scope):
        # 调整label列的位置到最后
        cols = list(data)
        try:
            cols.append(cols.pop(cols.index('label')))
            data = data_df.ix[:, cols]
        except:
            print("DataFrame need have col with name 'label'.")

        start = line_scope[0]
        end = line_scope[1]
        length = end - start + 1

        feature_num = data.shape[1] - 1
        label = data.ix[start:end, 'label'].astype('str')

        tmp = pd.Series([str(0)] * length)
        feature_all = tmp.str.cat(data.ix[start:end, all_cols[0]].astype('str'), sep=':')
        feature_all = label.str.cat(feature_all, sep=' ')

        for i in range(1, feature_num):
            tmp = pd.Series([str(i)] * length)
            feature_tmp = tmp.str.cat(data_df.ix[start:end, all_cols[i]].astype('str'), sep=':')
            feature_all = feature_all.str.cat(feature_tmp, sep=' ')
        return data, feature_all


