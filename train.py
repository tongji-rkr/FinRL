import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from IPython import display
from meta import config
from data_processor import DataProcessor
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import (
    StockTradingEnv,
)
from agents.stablebaselines3_models import DRLAgent
import os
from typing import List
from argparse import ArgumentParser
from meta import config
import pyfolio
from pyfolio import timeseries
import numpy as np

pd.options.display.max_columns = None

### data processing and feature engineering
TRAIN_START_DATE = "2018-01-02"
TRAIN_END_DATE = "2022-01-01"
TRADE_START_DATE = "2022-01-01"
TRADE_END_DATE = "2022-07-08"
TIME_INTERVAL = "1d"

p = DataProcessor(
    data_path='../Dataset/process_2018_2022_sp500_data_label.csv',#改数据集
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL,
)

# judge if the data is nan
isnan = p.dataframe.isnull().values.any()
print("data is nan : "+str(isnan))

# add_technical_indicator
#p.clean_data()
p.add_technical_indicator(config.INDICATORS)


### Split traning dataset

train = p.data_split(p.dataframe, TRAIN_START_DATE, TRAIN_END_DATE)

print(f"len(train.tic.unique()) : {len(train.tic.unique())}")

print(f"train.tic.unique() : {train.tic.unique()}")

print(f"train.head() : {train.head()}")

print(f"train.shape : {train.shape}")

stock_dimension = len(train.tic.unique())
state_space = stock_dimension * (len(config.INDICATORS) + 2) + 1
print(f"Stock Dimension : {stock_dimension}, State Space : {state_space}")

### Train

env_kwargs = {
    "model": "DDPG",
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 1000000,
    "buy_cost_pct": 6.87e-5,
    "sell_cost_pct": 1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": True,
    "hundred_each_trade": True
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
print(f"print(type(env_train)): {print(type(env_train))}")

if env_kwargs["model"] == "DDPG":
    # DDPG

    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {
        "batch_size": 256,
        "buffer_size": 50000,
        "learning_rate": 0.0005,
        "action_noise": "normal",
    }
    POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
    model_ddpg = agent.get_model(
        "ddpg", model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS
    )

    trained_ddpg = agent.train_model(
        model=model_ddpg, tb_log_name="ddpg", total_timesteps=100
    )
    # save model static
    trained_ddpg.save("ddpg_model")

    trade_model = trained_ddpg

elif env_kwargs["model"] == "A2C":
    # A2C

    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")

    trained_a2c = agent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=50000
    )
    # save model static
    trained_a2c.save("a2c_model")

    trade_mode = trained_a2c

elif env_kwargs["model"] == "PPO":
    # PPO

    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model("ppo")

    trained_ppo = agent.train_model(
        model=model_ppo, tb_log_name="ppo", total_timesteps=50000
    )
    # save model static
    trained_ppo.save("ppo_model")

    trade_model = trained_ppo


### Trade
trade = p.data_split(p.dataframe, TRADE_START_DATE, TRADE_END_DATE)
env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 1000000,
    "buy_cost_pct": 6.87e-5,
    "sell_cost_pct": 1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": False,
    "hundred_each_trade": True,
}
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

df_account_value, df_actions, rewards = DRLAgent.DRL_prediction(
    model=trade_model, environment=e_trade_gym
)

# plot rewards
import matplotlib.pyplot as plt
plt.xlabel("episode")
plt.ylabel("reward")
plt.plot(rewards)
plt.show()

# print sum of rewrad
total_reward = sum(rewards)
print('总收益为 : '+str(rewards))

df_actions.to_csv("action.csv", index=False)
print(f"df_actions: {df_actions}")

### Backtest

# matplotlib inline
plotter = ReturnPlotter(df_account_value, trade, TRADE_START_DATE, TRADE_END_DATE)
# plotter.plot_all()

plotter.plot()

# CSI 300
baseline_df = plotter.get_baseline("399300")


daily_return = plotter.get_return(df_account_value)
daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=daily_return,
    factor_returns=daily_return_base,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
print("==============DRL Strategy Stats===========")
print(f"perf_stats_all: {perf_stats_all}")


daily_return = plotter.get_return(df_account_value)
daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=daily_return_base,
    factor_returns=daily_return_base,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)
print("==============Baseline Strategy Stats===========")

print(f"perf_stats_all: {perf_stats_all}")
