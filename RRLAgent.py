import time
import numpy as np
import pandas as pd
import yaml
from datetime import datetime as dt
import matplotlib.pyplot as plt
from pandas_datareader import data
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S')


class BaseRRLTrader():
    def __init__(
            self,
            trading_periods: int,
            start_period: int,
            learning_rate: float,
            n_epochs: int,
            transaction_costs: float,
            input_size: int,
            added_features=True,
            SMA=False,
            epsilon_greedy=False,
            epsilon=0.5,
            bounds=3.0):
        """
        Initialisation of the RRLTrader Obj
        Args:
            trading_periods (int): the amount of days we add to the starting point
            start_period (int): the starting point from which we trade
            learning_rate (float): adaption rate between 0 and 1
            n_epochs (int): number of iterations
            transaction_costs (float): costs per transactions
            input_size (int): lookback period
            added_features (bool, optional):  Defaults to True.
            kind (str, optional): Reward Function. Defaults to "SR" (Sharpe Ratio).3
        """
        self.trading_periods = trading_periods
        self.lookback = input_size
        self.added_features = added_features
        if added_features:
            self.input_size = input_size + 18
        else:
            self.input_size = input_size
        self.start_date_trading = start_period
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.transaction_costs = transaction_costs
        self.all_t = None
        self.all_p = None
        self.time = None
        self.price = None
        self.returns = None
        self.SMA = SMA
        self.input_vector = np.zeros([trading_periods, self.input_size+2])
        self.action_space = np.zeros(trading_periods+1)
        self.dDSRdR = np.zeros(trading_periods)
        self.dRdF = np.zeros(trading_periods)
        self.dRdFp = np.zeros(trading_periods)
        self.dDSRdw = np.zeros([trading_periods, self.input_size+2])
        self.w = np.ones(self.input_size + 2)
        self.w_opt = np.ones(self.input_size + 2)
        self.epoch_training = np.empty(0)
        self.epsilon_greedy = epsilon_greedy
        if self.epsilon_greedy:
            self.epsilon = epsilon
            self.bounds = bounds

    def add_features(self):
        """
        Adding features to the object.
        - Simple Moving Averages (5 Day, 10 Day, 25 Day, 50 Day)
        - Upper and Lower Bollinger Bands - 2 Standard Deviations (5 Day, 10 Day, 25 Day, 50 Day)+
        - Moving Average Convergence Divergence 50 to 25 and 25 to 10
        - Momentum Factors
        """
        returns = np.array(list(self.input_df["RETURNS"])[::-1])

        self.sma5 = movingaverage(returns, 5)  # 1
        self.upper_bol_5 = self.sma5 + self.sma5 ** 2  # 2
        self.lower_bol_5 = self.sma5 - self.sma5 ** 2  # 3

        self.sma10 = movingaverage(returns, 10)  # 4
        self.upper_bol_10 = self.sma10 + self.sma10 ** 2  # 5
        self.lower_bol_10 = self.sma10 - self.sma10 ** 2  # 6

        self.sma25 = movingaverage(returns, 25)  # 7
        self.upper_bol_25 = self.sma25 + self.sma25 ** 2  # 8
        self.lower_bol_25 = self.sma25 - self.sma25 ** 2  # 9

        self.sma50 = movingaverage(returns, 50)  # 10
        self.upper_bol_50 = self.sma50 + self.sma50 ** 2  # 11
        self.lower_bol_50 = self.sma50 - self.sma50 ** 2  # 12

        # 13
        self.MACD_50_25 = self.sma25[:-
            (len(self.sma25)-len(self.sma50))] - self.sma50
        # 14
        self.MACD_25_10 = self.sma10[:-
            (len(self.sma10)-len(self.sma25))] - self.sma25
        # 15
        self.MACD_10_5 = self.sma5[:-
            (len(self.sma5)-len(self.sma10))] - self.sma10


    def upload_data(self, df:pd.DataFrame, start_date: str, end_date: str):
        """Fetches Data by Ticker &
        saves time, prices and returns a numpy arrays
        within the Agent object.
        """

        self.input_df = df

        all_t = np.array(list(self.input_df.index)[::-1])
        all_p = np.array(list(self.input_df["Adj Close"])[::-1])



        self.time = all_t[
            self.start_date_trading:
            self.start_date_trading +
            self.trading_periods + self.lookback+1]

        self.returns = all_p[
            self.start_date_trading:
            self.start_date_trading +
            self.trading_periods + self.lookback+1]



    def set_action_space(
            self,
            epsilon= 0.5):
        for i in range(self.trading_periods-1, -1, -1):
            self.feature = False
            self.input_vector[i] = np.zeros(self.input_size+2)
            self.input_vector[i][0] = 1.0
            self.input_vector[i][self.input_size+1] = self.action_space[i+1]
            for j in range(1, self.input_size+1, 1):
                if j > self.lookback:
                    if self.feature is False:

                        self.input_vector[i][j] = self.sma5[i]
                        self.input_vector[i][j+2] = self.lower_bol_5[i]
                        self.input_vector[i][j+1] = self.upper_bol_5[i]

                        self.input_vector[i][j+3] = self.sma10[i]
                        self.input_vector[i][j+4] = self.upper_bol_10[i]
                        self.input_vector[i][j+5] = self.lower_bol_10[i]

                        self.input_vector[i][j+6] = self.sma25[i]
                        self.input_vector[i][j+7] = self.upper_bol_25[i]
                        self.input_vector[i][j+8] = self.lower_bol_25[i]

                        self.input_vector[i][j+9] = self.sma50[i]
                        self.input_vector[i][j+10] = self.upper_bol_50[i]
                        self.input_vector[i][j+11] = self.lower_bol_50[i]

                        self.input_vector[i][j+12] = self.MACD_50_25[i]
                        self.input_vector[i][j+13] = self.MACD_25_10[i]
                        self.input_vector[i][j+14] = self.MACD_10_5[i]

                        self.input_vector[i][j+15] = self.Momentum_1M[i]
                        self.input_vector[i][j+16] = self.Momentum_3M[i]
                        self.input_vector[i][j+17] = self.Momentum_6M[i]
                        self.input_vector[i][j+18] = self.Momentum_1Y[i]
                        self.feature = True

                    else:
                        pass

                else:
                    self.input_vector[i][j] = self.returns[self.lookback+i-j]
            """
            if self.epsilon_greedy:
                if random.random() > epsilon:
                    self.action_space[i] = np.random.randint(-1,2)
                else:
                    self.action_space[i] = np.tanh(np.dot(self.w, self.input_vector[i]))
            else: """
            self.action_space[i] = np.tanh(
                np.dot(self.w, self.input_vector[i]))

    def calculate_action_returns(self):
        self.action_returns = (
            self.action_space[1:] * self.returns[:self.trading_periods] - self.transaction_costs * np.abs(-np.diff(self.action_space)))

    def calculate_cumulative_action_returns(self):
        self.sumR = np.cumsum(self.action_returns[::-1])[::-1]
        self.sumR2 = np.cumsum((self.action_returns**2)[::-1])[::-1]

    def RewardFunction(self):
        self.set_action_space()
        self.calculate_action_returns()
        self.calculate_cumulative_action_returns()
        self.A = self.sumR[0] / self.trading_periods
        self.B = self.sumR2[0] / self.trading_periods
        self.S = self.A / np.sqrt(self.B - self.A**2)
        self.dSdA = self.S * (1 + self.S**2) / self.A
        self.dSdB = -self.S**3 / 2 / self.A**2
        self.dAdR = 1.0 / self.trading_periods
        self.dBdR = 2.0 / self.trading_periods * self.action_returns
        self.dRdF = - self.transaction_costs * \
            np.sign(-np.diff(self.action_space))
        self.dRdFp = self.returns[:self.trading_periods] + self.transaction_costs * np.sign(-np.diff(self.action_space))
        self.dFdw = np.zeros(self.input_size+2)
        self.dFpdw = np.zeros(self.input_size+2)
        self.dSdw = np.zeros(self.input_size+2)
        for i in range(self.trading_periods-1, -1, -1):
            if i != self.trading_periods-1:
                self.dFpdw = self.dFdw.copy()
            self.dFdw = (1 - self.action_space[i]**2) * (self.input_vector[i] + self.w[self.input_size+1] * self.dFpdw)
            self.dSdw += (self.dSdA * self.dAdR + self.dSdB * self.dBdR[i]) * (
                self.dRdF[i] * self.dFdw + self.dRdFp[i] * self.dFpdw)

    def fit(self):
        pre_epoch_times = len(self.epoch_training)
        self.RewardFunction()
        print("Epoch loop start. Initial Sharpe ratio : " + str(self.S) + ".")
        self.S_opt = self.S

        tic = time.process_time()
        for e_index in range(self.n_epochs):
            self.RewardFunction()
            if self.S > self.S_opt:
                self.S_opt = self.S
                self.w_opt = self.w.copy()
            self.epoch_training = np.append(self.epoch_training, self.S)
            if e_index < 0:
                self.w = np.random.rand(self.input_size+2)
            elif e_index == 0:
                self.w = self.w_opt
            if e_index > 0:
                self.update_weight()
            if e_index % 100 == 100-1:
                toc = time.clock()
                logging.info("INFO: Epoch {}/{} | Optimal SR: {} | Current SR {} | Elapsed time {} sec.".format(str(e_index + pre_epoch_times + 1), str(self.n_epochs + pre_epoch_times), str(self.S_opt),str(self.S),str(toc - tic)))
        toc = time.clock()
        print("Epoch: " + str(e_index + pre_epoch_times + 1) + "/" + str(self.n_epochs + pre_epoch_times) + \
              ". Sharpe ratio: " + str(self.S) + ". Elapsed time: " + str(toc-tic) + " sec.")
        self.w = self.w_opt.copy()
        self.RewardFunction()
        print("Epoch loop end. Optimized Sharpe ratio is " + str(self.S_opt) + ".")

    def update_weight(
            self):
        if self.epsilon_greedy:
            if random.random() > self.epsilon:
                self.w = np.random.uniform(
                    -1 * self.bounds,
                    self.bounds,
                    self.input_size+2)
                self.RewardFunction()
                if self.S > self.S_opt:
                    self.S_opt = self.S
                    self.w_opt = self.w.copy()
                else:
                    self.w = self.w_opt
            else:
                self.w += self.learning_rate * self.dSdw

        else:
            self.w += self.learning_rate * self.dSdw



def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
