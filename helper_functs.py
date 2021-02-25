"""
Helper Functions for Big Data 1. 
"""

import pandas as pd
import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


def create_stock_df(no, sources):
    new_df = pd.DataFrame(index=sources[0].index)
    for i in sources:
        if no in i.columns:
            new_df[i.name] = i[no]
    return new_df


def get_lags(df, n_lags):
    for col_name in df.columns:
        for l in range(1,n_lags+1):
            df[f'{col_name}_LAG_{l}']= df [col_name].shift(l)
    return df

def x_y_split(df):
    x = df.drop(["FLOWS", "RETURNS"], axis=1)
    y = df[['RETURNS']]
    return x,y

def extend_variables(no,x_small,returns,flows):
    cl_returns = returns.drop([no],axis=1)
    cl_flows = flows.drop([no], axis=1)
    cl_returns.columns = [f"{s}_RETURNS_LAG_1" for s in cl_returns.columns]
    cl_flows.columns = [f"{s}_FLOWS_LAG_1"   for s in cl_flows.columns]
    x_small[cl_returns.columns]  = cl_returns.shift(1)
    x_small[cl_flows.columns]  = cl_flows.shift(1)
    return x_small


def Rolling_ML(window_size, df, model, hyperparameters={}, progession_param=0):
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

