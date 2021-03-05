"""
Helper Functions for Big Data 1. 
"""

import pandas as pd
import copy
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA

def create_stock_df(permno:int, sources:list)->pd.DataFrame:
    """
    Create a pd.Dataframe given PERMNO and other df columns in a list.
    """
    new_df = pd.DataFrame(index=sources[0].index)
    for i in sources:
        if permno in i.columns:
            new_df[i.name] = i[permno]
    return new_df


def get_lags(df:pd.DataFrame, n_lags:int) -> pd.DataFrame:
    """
    Adds n lags for every column to a pd.DataFrame.
    """
    for col_name in df.columns:
        for l in range(1,n_lags+1):
            df[f'{col_name}_LAG_{l}']= df [col_name].shift(l)
    return df

def x_y_split(df:pd.DataFrame) -> pd.DataFrame:
    """
    Splits pd.DataFrame into known and unknown parts at any point. 
    """
    x = df.drop(["FLOWS", "RETURNS"], axis=1)
    y = df[['RETURNS']]
    return x,y


def extend_variables(
    no:int, x_small:pd.DataFrame, 
    returns:pd.DataFrame, flows:pd.DataFrame, 
    pca_n_components:int=0)-> pd.DataFrame:
    cl_returns = returns.drop([no], axis=1).shift(1)
    cl_flows = flows.drop([no], axis=1).shift(1)

    if pca_n_components != 0:
        pca = PCA(n_components=pca_n_components)
        cl_returns.dropna(inplace=True)
        pca_returns = pca.fit_transform(cl_returns)
        principal_returns_df = pd.DataFrame(data=pca_returns, columns=[
                                            f"PCA_{s}_RETURNS_LAG_1" for s in range(pca_n_components)], index=cl_returns.index)

        cl_flows.dropna(inplace=True)
        pca_flows = pca.fit_transform(cl_flows)
        principal_flows_df = pd.DataFrame(data=pca_flows, columns=[
            f"PC_{s}_FLOWS_LAG_1" for s in range(pca_n_components)], index=cl_flows.index)

        x_small[principal_flows_df.columns] = principal_flows_df
        x_small[principal_returns_df.columns] = principal_returns_df
        return x_small

    else:
        cl_flows.columns = [f"{s}_FLOWS_LAG_1" for s in cl_flows.columns]
        cl_returns.columns = [f"{s}_RETURNS_LAG_1" for s in cl_returns.columns]
        x_small[cl_returns.columns] = cl_returns
        x_small[cl_flows.columns] = cl_flows

        return x_small


def Rolling_ML(window_size:int, df:pd.DataFrame, model, hyperparameters={}, progession_param=0):
    df_x, df_y = x_y_split(df)
    stock_ML = copy.deepcopy(df_y)
    stock_ML["PRED"] = None
    if progession_param != 0:
        step_size = int(window_size*progession_param)
    else:
        step_size = 1
    for i, j in enumerate(range(window_size, len(stock_ML)), step_size):
        if len(hyperparameters.keys()) != 0:
            cv = RandomizedSearchCV(model, hyperparameters, random_state=0)
            clf = cv.fit(df_x[i:j], df_y[i:j])
        else:
            clf = model.fit(df_x[i:j], df_y[i:j])
        stock_ML.loc[j:j+step_size,
                     'PRED'] = np.array(clf.predict(df_x[j:j+step_size]))
    return stock_ML


def Normal_ML(window_size, df, model, hyperparameters={}):
    df_x, df_y = x_y_split(df)
    X_train, y_train = df_x[:int(len(df_x)*0.7)], df_y[:int(len(df_y)*0.7)]
    X_test, y_test = df_x[int(len(df_x)*0.7):], df_y[int(len(df_y)*0.7):]

    if len(hyperparameters.keys()) != 0:
        cv = RandomizedSearchCV(model, hyperparameters, random_state=0)
        clf = cv.fit(X_train, y_train)
    else:
        clf = model.fit(X_train, y_train)

    y_test['PRED'] = clf.predict(X_test)
    return y_test



def add_features(df):
    """
    Generally technical indicators.  
    """
    df['SMA5'] = df["RETURNS"].rolling(5).mean()
    df['SMA5_VOL'] = df["RETURNS"].rolling(5).std()
    df["UPPER_BOL_5"] = df['SMA5'] + df['SMA5_VOL'] ** 2
    df["LOWER_BOL_5"] = df['SMA5'] - df['SMA5_VOL'] ** 2
    
    df['SMA10'] = df["RETURNS"].rolling(10).mean()
    df['SMA10_VOL'] = df["RETURNS"].rolling(10).std()
    df["UPPER_BOL_10"] = df['SMA10'] + df['SMA10_VOL'] ** 2
    df["LOWER_BOL_10"] = df['SMA10'] - df['SMA10_VOL'] ** 2

    df['SMA25'] = df["RETURNS"].rolling(25).mean()
    df['SMA25_VOL'] = df["RETURNS"].rolling(25).std()
    df["UPPER_BOL_25"] = df['SMA25'] + df['SMA25_VOL'] ** 2
    df["LOWER_BOL_25"] = df['SMA25'] - df['SMA25_VOL'] ** 2

    df['SMA50'] = df["RETURNS"].rolling(50).mean()
    df['SMA50_VOL'] = df["RETURNS"].rolling(50).std()
    df["UPPER_BOL_50"] = df['SMA50'] + df['SMA50_VOL'] ** 2
    df["LOWER_BOL_50"] = df['SMA50'] - df['SMA50_VOL'] ** 2
    df.dropna(inplace=True)
    return df


def get_threshold(value, tau=0.01):
    if abs(value) > tau:
        return np.sign(value)
    else:
        return 0


def make_market_state(series: pd.Series, rolling=5, tau=0.01):
    if rolling !=0:
        forward_sma = series.rolling(rolling).mean().shift(-(rolling-1))
        forward_sma.iloc[-(rolling-1):] = series.iloc[-(rolling-1):]
        market_state = forward_sma.apply(get_threshold, tau=tau)
    else: 
        market_state = forward_sma.apply(get_threshold, tau=tau)
    return market_state
