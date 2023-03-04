import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import stockstats
import talib
import copy
from copy import deepcopy

class DataProcessor:
    def __init__(
        self,
        data_path: str,
        start_date: str,
        end_date: str,
        time_interval: str
    ):
        self.data_path = data_path
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.dataframe=pd.read_csv(data_path,encoding='utf-8',index_col=0)
        if "dt" in self.dataframe.columns.values.tolist():
            self.dataframe = self.dataframe.rename(columns={"dt": "date"})
        if "kdcode" in self.dataframe.columns.values.tolist():
            self.dataframe = self.dataframe.rename(columns={"kdcode": "tic"})
        self.ticker_list=self.dataframe.tic.unique()
        #print(self.dataframe.head())
    
    def clean_data(self):
        dfc = copy.deepcopy(self.dataframe)
        dfcode = pd.DataFrame(columns=["tic"])
        dfdate = pd.DataFrame(columns=["date"])
        print(dfc.head())
        dfcode.tic = dfc.tic.unique()
        dfdate.date = dfc.date.unique()
        dfdate.sort_values(by="date", ascending=False, ignore_index=True, inplace=True)

        # the old pandas may not support pd.merge(how="cross")
        try:
            df1 = pd.merge(dfcode, dfdate, how="cross")
        except:
            print("Please wait for a few seconds...")
            df1 = pd.DataFrame(columns=["tic", "date"])
            for i in range(dfcode.shape[0]):
                for j in range(dfdate.shape[0]):
                    df1 = df1.append(
                        pd.DataFrame(
                            data={
                                "tic": dfcode.iat[i, 0],
                                "date": dfdate.iat[j, 0],
                            },
                            index=[(i + 1) * (j + 1) - 1],
                        )
                    )

        df2 = pd.merge(df1, dfc, how="left", on=["tic", "date"])

        # back fill missing data then front fill
        df3 = pd.DataFrame(columns=df2.columns)
        for i in self.ticker_list:
            df4 = df2[df2.tic == i].fillna(method="bfill").fillna(method="ffill")
            df3 = pd.concat([df3, df4], ignore_index=True)

        df3 = df3.fillna(0)

        # reshape dataframe
        df3 = df3.sort_values(by=["date", "tic"]).reset_index(drop=True)

        print("Shape of DataFrame: ", df3.shape)

        self.dataframe = df3

    def add_technical_indicator(
        self, tech_indicator_list: List[str], select_stockstats_talib: int = 0
    ):
        assert select_stockstats_talib in {0, 1}
        print("tech_indicator_list: ", tech_indicator_list)
        if select_stockstats_talib == 0:  # use stockstats
            stock = stockstats.StockDataFrame.retype(self.dataframe)
            unique_ticker = stock.tic.unique()
            # print("unique_ticker: ", unique_ticker)
            for indicator in tech_indicator_list:
                print("indicator: ", indicator)
                indicator_df = pd.DataFrame()
                for i in range(len(unique_ticker)):
                    try:
                        temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                        temp_indicator = pd.DataFrame(temp_indicator)
                        temp_indicator["tic"] = unique_ticker[i]
                        temp_indicator["date"] = self.dataframe[
                            self.dataframe.tic == unique_ticker[i]
                        ]["date"].to_list()
                        indicator_df = pd.concat(
                            [indicator_df, temp_indicator],
                            axis=0,
                            join="outer",
                            ignore_index=True,
                        )
                    except Exception as e:
                        print(e)
                if not indicator_df.empty:
                    self.dataframe = self.dataframe.merge(
                        indicator_df[["tic", "date", indicator]],
                        on=["tic", "date"],
                        how="left",
                    )
        else:  # use talib
            final_df = pd.DataFrame()
            for i in self.dataframe.tic.unique():
                tic_df = self.dataframe[self.dataframe.tic == i]
                (
                    tic_df.loc["macd"],
                    tic_df.loc["macd_signal"],
                    tic_df.loc["macd_hist"],
                ) = talib.MACD(
                    tic_df["close"],
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9,
                )
                tic_df.loc["rsi"] = talib.RSI(tic_df["close"], timeperiod=14)
                tic_df.loc["cci"] = talib.CCI(
                    tic_df["high"],
                    tic_df["low"],
                    tic_df["close"],
                    timeperiod=14,
                )
                tic_df.loc["dx"] = talib.DX(
                    tic_df["high"],
                    tic_df["low"],
                    tic_df["close"],
                    timeperiod=14,
                )
                final_df = pd.concat([final_df, tic_df], axis=0, join="outer")
            self.dataframe = final_df

        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        time_to_drop = self.dataframe[self.dataframe.isna().any(axis=1)].date.unique()
        self.dataframe = self.dataframe[~self.dataframe.date.isin(time_to_drop)]
        print("Succesfully add technical indicators")

    def data_split(self, df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data