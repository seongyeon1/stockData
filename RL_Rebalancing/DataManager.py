import pandas as pd
import numpy as np
from collections import defaultdict

Features_Raw = ["Open", "High", "Low", "Close", "Volume", "Price"]


def get_data(path,
             train_date_start=None, train_date_end=None,
             test_date_start=None, test_date_end=None):
    data = pd.read_csv(path, thousands=",", converters={"Date": lambda x: str(x)})
    data = data.replace(0, np.nan)
    data = data.fillna(method="bfill")
    data.insert(data.shape[1], "Price", data["Close"].values)
    data, _ = get_scaling(data)

    train_date_start = data["Date"].iloc[0] if train_date_start is None else train_date_start
    train_date_end = data["Date"].iloc[-1] if train_date_end is None else train_date_end
    test_date_start = data["Date"].iloc[0] if test_date_start is None else test_date_start
    test_date_end = data["Date"].iloc[-1] if test_date_end is None else test_date_end

    train_data = data[(data["Date"] >= train_date_start) & (data["Date"] <= train_date_end)]
    test_data = data[(data["Date"] >= test_date_start) & (data["Date"] <= test_date_end)]

    train_data = train_data.set_index("Date", drop=True)
    test_data = test_data.set_index("Date", drop=True)

    train_data = train_data.astype("float32")
    test_data = test_data.astype("float32")
    return train_data.loc[:,Features_Raw], test_data.loc[:,Features_Raw]


def get_data_tensor(path_list,
                    train_date_start=None, train_date_end=None,
                    test_date_start=None, test_date_end=None):

    for path in path_list:
        train_data, test_data = get_data(path,
                                         train_date_start, train_date_end,
                                         test_date_start, test_date_end)

        if path == path_list[0]:
            common_date_train = set(train_data.index.unique())
            common_date_test = set(test_data.index.unique())
        else:
            common_date_train = common_date_train & set(train_data.index.unique())
            common_date_test = common_date_test & set(test_data.index.unique())

    common_date_train = list(common_date_train)
    common_date_test = list(common_date_test)

    common_date_train.sort()
    common_date_test.sort()

    for path in path_list:
        train_data_, test_data_ = get_data(path,
                                         train_date_start, train_date_end,
                                         test_date_start, test_date_end)

        train_data_ = train_data_[train_data_.index.isin(common_date_train)].to_numpy()
        test_data_ = test_data_[test_data_.index.isin(common_date_test)].to_numpy()

        train_data_ = train_data_[:, :, np.newaxis]
        test_data_ = test_data_[:, :, np.newaxis]

        if path == path_list[0]:
            train_data = train_data_
            test_data = test_data_
        else:
            train_data = np.concatenate([train_data, train_data_], axis=-1)
            test_data = np.concatenate([test_data, test_data_], axis=-1)

    print("-------------------------------------")
    print(f"학습 데이터 시작 거래일:{common_date_train[0]}")
    print(f"학습 데이터 마지막 거래일:{common_date_train[-1]}")
    print(f"테스트 데이터 시작 거래일:{common_date_test[0]}")
    print(f"테스트 데이터 마지막 거래일:{common_date_test[-1]}")
    print("-------------------------------------")
    return train_data, test_data


def get_scaling(data):
    feature_names = list(data.columns)
    not_scaling = ["MACD", "MACDsignal", "MACDoscillator", "Price", "Date"]
    info = defaultdict(float)

    for name in not_scaling:
        if name in data.columns:
            feature_names.remove(name)
    for name in feature_names:
        feature_data = data.loc[:, name]
        feature_mean = feature_data.mean()
        feature_std = feature_data.std()
        data.loc[:, name] = (data.loc[:, name] - feature_mean) / feature_std
        info[name] = (feature_mean, feature_std)
    return data, info

