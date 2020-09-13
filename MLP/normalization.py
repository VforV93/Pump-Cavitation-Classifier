import numpy as np
import pandas as pd
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

seed(42)  # keras seed fixing
plt.style.use('ggplot')

path_prefix = "~ Documents/"  # data-raw path
path = "data-raw/DS1/FILTERED/"
test_path = "data-raw/DS2/FILTERED/"

"""
Load Data
If OK, IN or STAND are None ignore it.
Load all the files in OK, IN and STAND. Split individually before and then merge them.
"""
def load_data(path, w, OK=None, IN=None, STAND=None, train_perc=0.75):
    dfs = list()

    # load OK dataset
    temp = list()
    if OK is not None:
        for name in OK:
            temp.append(pd.read_csv(path_prefix + path + name + w + ".csv", sep='\t'))
        if len(temp) > 0:
            dfs.append(pd.concat(temp, sort=False, ignore_index=True))

    # load IN dataset
    temp = list()
    if IN is not None:
        for name in IN:
            temp.append(pd.read_csv(path_prefix + path + name + w + ".csv", sep='\t'))
        if len(temp) > 0:
            dfs.append(pd.concat(temp, sort=False, ignore_index=True))

    # load STAND dataset
    temp = list()
    if STAND is not None:
        for name in STAND:
            temp.append(pd.read_csv(path_prefix + path + name + w + ".csv", sep='\t'))
        if len(temp) > 0:
            dfs.append(pd.concat(temp, sort=False, ignore_index=True))

    ret = list()
    ret_val = list()
    # Data preprocessing
    for i, df_split in enumerate(dfs):
        dim = int(len(df_split) * train_perc)
        splitted = np.split(df_split, [dim])
        if len(splitted[0]) > 0:
            ret.append(splitted[0])
        if len(splitted[1]) > 0:
            ret_val.append(splitted[1])

    df = None
    df_val = None
    for i in range(len(ret)):
        if df is not None:
            df = df.append(ret[i], ignore_index=True)
        else:
            df = ret[i]

    for i in range(len(ret_val)):
        if df_val is not None:
            df_val = df_val.append(ret_val[i], ignore_index=True)
        else:
            df_val = ret_val[i]

    return df, df_val  # training dataframe, validation dataframe


"""
online normalization on all entire dataframe
"""
def normalization(dataframe):
    dataframe['cycle_norm'] = dataframe['cycle']
    cols_normalize = dataframe.columns.difference(['time', 'cycle', 'Class'])
    min_max_scaler = MinMaxScaler((-1, 1))

    norm_df = pd.DataFrame(min_max_scaler.fit_transform(dataframe[cols_normalize]),
                           columns=cols_normalize,
                           index=dataframe.index)

    join_df = dataframe[dataframe.columns.difference(cols_normalize)].join(norm_df)

    # One hot encoding of the Class column
    join_df['Class'] = pd.Categorical(join_df['Class'])
    dfDummies = pd.get_dummies(join_df['Class'], prefix='category')
    join_df = pd.concat([join_df, dfDummies], axis=1)
    return join_df.reindex(columns=join_df.columns)


"""
Normalization based on a sliding temporal window.
es: window = 75k = 3 seconds of signal.
We start to normalize from the beginning to the end of the dataframe, 3 seconds at a time 
"""
def window_normalization(dataframe: pd.DataFrame, window):
    dataframe['cycle_norm'] = dataframe['cycle']
    cols_normalize = dataframe.columns.difference(['time', 'cycle', 'Class'])
    min_max_scaler = MinMaxScaler((-1, 1))

    # df_val.append(ret_val[i], ignore_index=True)
    temp = None
    print("starting...")
    for di in range(0, len(dataframe), window):
        min_max_scaler.fit(dataframe.iloc[di:di + window][cols_normalize])
        if temp is None:
            temp = min_max_scaler.transform(dataframe.iloc[di:di + window][cols_normalize])
        else:
            temp = np.concatenate((temp, (min_max_scaler.transform(dataframe.iloc[di:di + window][cols_normalize]))))

    # Normalization of the last bit of remaining data
    # min_max_scaler.fit(dataframe.iloc[di + window:][cols_normalize])
    print("end!")
    norm_df = pd.DataFrame(temp, columns=cols_normalize, index=dataframe.index)
    join_df = dataframe[dataframe.columns.difference(cols_normalize)].join(norm_df)

    # One hot encoding of the Class column
    join_df['Class'] = pd.Categorical(join_df['Class'])
    dfDummies = pd.get_dummies(join_df['Class'], prefix='category')
    join_df = pd.concat([join_df, dfDummies], axis=1)

    return join_df.reindex(columns=join_df.columns)


# for each windows specified, apply the sliding temporal normalization and save the data to the file system
def normalize_split_save(df, path, files_name, ws=[25000, 75000, 150000]):
    for w in ws:
        df_norm = window_normalization(df, w)

        df_OK = df_norm.loc[df_norm['category_OK'] == 1]
        df_IN = df_norm.loc[df_norm['category_IN'] == 1]
        df_STAND = df_norm.loc[df_norm['category_STANDING'] == 1]

        w_text = str(int(w/1000)) + 'k.csv'

        df_OK.to_csv(path_prefix + path + 'norm/' + files_name[0] + '_' + w_text, sep='\t', index=False, encoding='utf-8')
        df_IN.to_csv(path_prefix + path + 'norm/' + files_name[1] + '_' + w_text, sep='\t', index=False, encoding='utf-8')
        df_STAND.to_csv(path_prefix + path + 'norm/' + files_name[2] + '_' + w_text, sep='\t', index=False, encoding='utf-8')


# we Load the filtered dataset with w5 and w15 for the sliding temporal normalization (training and testing)
w_filt = [5, 15]
for wt in w_filt:
    OK_csv = ["OK1_FILTERED_w", "OK2_FILTERED_w"]
    IN_csv = ["IN1_FILTERED_w"]
    STAND_csv = ["STANDING1_FILTERED_w"]

    OK_test_csv = ["OK_FILTERED_w"]
    IN_test_csv = ["IN_FILTERED_w"]
    STAND_test_csv = ["STANDING1_FILTERED_w"]

    df, _ = load_data(path, str(wt), OK_csv, IN_csv, STAND_csv, 1)
    df_test, _ = load_data(test_path, str(wt), OK_test_csv, IN_test_csv, STAND_test_csv, 1)

    normalize_split_save(df, path, ["OK1_2_w"+str(wt)+"_norm", "IN1_w"+str(wt)+"_norm", "STANDING1_w"+str(wt)+"_norm"])
    normalize_split_save(df_test, test_path, ["OK_w"+str(wt)+"_norm", "IN_w"+str(wt)+"_norm", "STANDING1_w"+str(wt)+"_norm"])

ws=[25000, 75000, 150000]
OK_csv = ["OK3_FILTERED_w", "OK4_FILTERED_w"]
IN_csv = None
STAND_csv = ["STANDING2_FILTERED_w"]

# as before for the OK3, OK4 and STANDING2
w_filt = [5, 15]
for wt in w_filt:
    df_ok, _ = load_data(path, str(wt), OK_csv, None, None, 1)
    df_stand, _ = load_data(path, str(wt), None, None, STAND_csv, 1)

    for w in ws:
        w_text = str(int(w/1000)) + 'k.csv'

        df_ok_norm = window_normalization(df_ok, w)
        df_ok_norm['category_IN'] = 0
        df_ok_norm['category_STANDING'] = 0
        df_ok_norm.to_csv(path_prefix + path + "norm/OK3_4_w"+str(wt)+"_norm" + '_' + w_text, sep='\t', index=False, encoding='utf-8')

        df_stand_norm = window_normalization(df_stand, w)
        df_stand_norm['category_OK'] = 0
        df_stand_norm['category_IN'] = 0
        df_stand_norm.to_csv(path_prefix + path + "norm/STANDING2_w" + str(wt) + "_norm" + '_' + w_text, sep='\t', index=False, encoding='utf-8')


OK_test_csv = None
IN_test_csv = None
STAND_test_csv = ["STANDING2_FILTERED_w", "STANDING3_FILTERED_w", "STANDING4_FILTERED_w"]
STAND_test_out = ["STANDING2_w", "STANDING3_w", "STANDING4_w"]
# as before for the last testing STANDING data (STANDING2, STANDING3, STANDING3)
for i, std_filt in enumerate(STAND_test_csv):
    for wt in w_filt:
        df_stand, _ = load_data(test_path, str(wt), None, None, [std_filt], 1)

        for w in ws:
            w_text = str(int(w/1000)) + 'k.csv'
            df_stand_norm = window_normalization(df_stand, w)
            df_stand_norm['category_OK'] = 0
            df_stand_norm['category_IN'] = 0
            df_stand_norm.to_csv(path_prefix + test_path + "norm/"+STAND_test_out[i] + str(wt) + "_norm" + '_' + w_text, sep='\t', index=False, encoding='utf-8')
