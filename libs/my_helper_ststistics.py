import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


# CI are frequently calculated at a confidence level of 95%.
def get_cf_bootstraping(func, data_pred, data_gt, sampling_times=500, cf_level=0.95):

    data_pred = np.array(data_pred)
    data_gt = np.array(data_gt)
    samples_num = len(data_gt)

    list_sample_result = []
    for i in range(sampling_times):
        # print(f'bootstrap sample times:{i}')
        index_arr = np.random.randint(0, samples_num, size=samples_num)
        sample_result = func(data_pred[index_arr], data_gt[index_arr])
        list_sample_result.append(sample_result)

    a = 1 - cf_level
    k1 = int(sampling_times * a / 2)
    k2 = int(sampling_times * (1 - a / 2))
    auc_sample_arr_sorted = sorted(list_sample_result)
    lower = auc_sample_arr_sorted[k1]
    higher = auc_sample_arr_sorted[k2]

    return lower, higher


def cal_rmse(predictions, targets):
    return sqrt(mean_squared_error(predictions, targets))

