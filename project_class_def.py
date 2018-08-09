"""
Group 6:
    authors: Ang Li,           17203382
             Shuhong Jiang,    15202005
             Suohuijia Wang,   17200170
"""


# -*- coding: UTF-8 -*-
import os
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import time
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
from pprint import pprint

# import Machine Learning Module
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# import general-purpose function from other files (to be more clear)
from dir_info import data_dir, buffer_dir
from model_config import nonlinear_model_config, linear_model_config
from utils import read_exist_companylist, write_company_name_list, exp_moving_average, read_exist_data

# all graphs displayed in the ggplot style 
# General setting
style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# Create dictionary automatically
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(buffer_dir):
    os.makedirs(buffer_dir)


class Financial_Analysis_Basis(object):
    def __init__(self):
        self.__initial_company_names_list()

    def __initial_company_names_list(self):
        """ TODO - first step- initial company name list, private properties, cannot be inherited"""
        if not os.path.exists(os.path.join(buffer_dir, "companylist.txt")):
            self.company_names_list = []
        else:
            self.company_names_list = read_exist_companylist(display=False)

    def _refresh_company_names_list(self):
        self.company_names_list = read_exist_companylist(display=False)

    def get_data_online(self, company_symbol, start_date, end_date):
        """ TODO - download data from Yahoo Finance"""
        t1 = time.strptime(start_date, "%Y-%m-%d")
        y1, m1, d1 = t1[0:3]
        start = dt.datetime(y1, m1, d1)
        t2 = time.strptime(end_date, "%Y-%m-%d")
        y2, m2, d2 = t2[0:3]
        end = dt.datetime(y2, m2, d2)
        df0 = web.DataReader(company_symbol, 'yahoo', start, end)
        df0.to_csv(os.path.join(data_dir, '{}.csv'.format(company_symbol)))
        print("%s data download done! You can check it in your 'Data' folder." % (company_symbol))
        return df0

    def _quantiles(self, df, a):  # 前面有个_说明在类里面可以用，内部函数，在外面不能用。
        """ TODO - descriptive_analytics(quantiles), protected properties, can be inherited"""
        quantile_data = df.quantile(a)
        return quantile_data

    def descriptive_analytics(self, company_symbol=None):
        """ TODO - descriptive_analytics, Menu Interface with choice """
        print("Here are existed companies: ")
        _ = read_exist_companylist(display=True)

        if company_symbol is None:
            company_symbol = input("Please enter the symbols your selected company:")

            if company_symbol not in self.company_names_list:
                print('%s is not in the existing company list, before you analyse it you need to download its data firstly...' % (company_symbol))
                company_symbol = company_symbol.upper()     # convert any input into upper case
                start_date = input('Please enter a start time, as format(yyyy-mm-dd): ')
                end_date = input('Please enter a end time, as format(yyyy-mm-dd): ')
                df = self.get_data_online(company_symbol=company_symbol, start_date=start_date, end_date=end_date)
            else:
                df = read_exist_data(company_symbol)
        else:
            if company_symbol not in self.company_names_list:
                print('%s is not in the existing company list, before you analyse it you need to download its data firstly...' % (company_symbol))
                company_symbol = company_symbol.upper()
                start_date = input('enter a start time, as format(yyyy-mm-dd): ')
                end_date = input('enter the end time, as format(yyyy-mm-dd): ')
                df = self.get_data_online(company_symbol=company_symbol, start_date=start_date, end_date=end_date)
            else:
                df = read_exist_data(company_symbol)

        print('You select %s to analyse!' % (company_symbol))
        print("Welcome to the descriptive analytics system.")
        print("1.Max\n2.Min\n3.Count\n4.Mean\n5.Variance\n6.Standard Deviation\n"
              "7.The date of the minimum for every column\n8.The date of the maximum for every column\n9.Quantiles\n10.Quit")
        choice = input("Please enter your choose option: ")
        while choice != "10":
            if choice == "1":
                print("The maximum for every column is\n{}.".format(df.max()))
            elif choice == '2':
                print("The minimum for every column is\n{}.".format(df.min()))
            elif choice == '3':
                print("This company has {} data in chosen period.".format(df['Open'].count()))
            elif choice == '4':
                print("The mean for every column is\n{}.".format(df.mean()))
            elif choice == '5':
                print("The variance for every column is\n{}.".format(df.var()))
            elif choice == '6':
                print("The standard deviation for every column is\n{}.".format(df.std()))
            elif choice == '7':
                print("The date of the minimum for every column is\n{}.".format(df.idxmin()))
            elif choice == '8':
                print("The date of the maximum for every column is\n{}.".format(df.idxmax()))
            elif choice == '9':
                print("Enter a certain percentage as an float" )
                a = float(input("Please ensure the number to be in the interval [0, 1]:  "))
                quantile_data = self._quantiles(df, a)
                print("The {} quantile for every column is\n{}".format('%.2f%%' % (a*100), quantile_data))
            else:
                print("Wrong choice, please try again.")
            print("1.Max\n2.Min\n3.Count\n4.Mean\n5.Variance\n6.Standard Deviation\n"
              "7.The date of the minimum for every column\n8.The date of the maximum for every column\n9.Quantiles\n10.Quit")
            choice = input("Please enter your choose option:")

    def plot_Kline_MA(self, company_symbol=None, start_date=None, end_date=None, moving_windows=[5,10,20,30]):
        """ TODO - Kline with 4 Moving Average, """
        print("Here are existed companies: ")
        _ = read_exist_companylist(display=True)

        if company_symbol is None:
            company_symbol = input("Please enter the symbols your selected company: ")

            if company_symbol not in self.company_names_list:
                print('%s is not in the existing company list, before you analyse it you need to download its data firstly...' % (company_symbol))
                company_symbol = company_symbol.upper()
                start_date = input('Please enter a start time, as format(yyyy-mm-dd): ')
                end_date = input('Please enter the end time, as format(yyyy-mm-dd): ')
                df = self.get_data_online(company_symbol=company_symbol, start_date=start_date, end_date=end_date)
            else:
                df = read_exist_data(company_symbol)
        else:
            if company_symbol not in self.company_names_list:
                print('%s is not in the existing company list, before you analyse it you need to download its data firstly...' % (company_symbol))
                company_symbol = company_symbol.upper()
                start_date = input('Please enter a start time, as format(yyyy-mm-dd): ')
                end_date = input('Please enter a end time, as format(yyyy-mm-dd): ')
                df = self.get_data_online(company_symbol=company_symbol, start_date=start_date, end_date=end_date)
            else:
                df = read_exist_data(company_symbol)

        print('You select %s to plot!' % (company_symbol))

        is_resample = input('\ncurrent data is daily chart, do you want to resample to generate a lower frequency K-lines, , format as Y/N:')
        if is_resample.lower() == 'y' or is_resample.lower() == 'yes':
            days = input("enter a time, format as 10D: ")
        else:
            days = '1D'

        # Set title
        title = '%s - %s-%say K-line' % (company_symbol, days[:-1], days[-1])

        if start_date is not None and end_date is not None:
            # 取的是所选择时间段内的数据，因为按照时间大小来算，大于start_time就是在它之后的数据
            # 所以实际上，就是选择原始数据中间的某一段数据。
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        elif start_date is not None and end_date is None:
            df = df[df.index >= pd.to_datetime(start_date)]
        elif start_date is None and end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        # processing data
        # adjust OHL data according to "adj close"
        # 这就是画K线图的规则，要乘以一个系数。
        adj_factor = df['Adj Close'] / df['Close']
        df['Open'] = df['Open'] * adj_factor
        df['High'] = df['High'] * adj_factor
        df['Low'] = df['Low'] * adj_factor

        if days == '1D':
            df_ohlc = df[['Open','High','Low','Adj Close']]
            df_ohlc.columns = ['open', 'high', 'low', 'close']
            df_volume = df['Volume']
            pass
        else:
            # resampling
            # 开盘价就是第一天的价格
            df_ohlc = pd.DataFrame(df['Open'].resample(days).first(), columns=['Open'])
            # 最高价就是这几天的最高价
            df_ohlc['high'] = df['High'].resample(days).max()
            # 最低价就是这几天的最低价
            df_ohlc['low'] = df['Low'].resample(days).min()
            # 收盘价就是最后一天的价格，但是用的是调整后的收盘价，可能是规则。
            df_ohlc['close'] = df['Adj Close'].resample(days).last()
            # 添加列名
            df_ohlc.columns = ['open','high','low','close']
            # 体量就是这几天的总交易量
            df_volume = df['Volume'].resample(days).sum()
            # Drop NAN 去掉空值，并更新数据
            df_ohlc.dropna(inplace=True)
            df_volume.dropna(inplace=True)

        # reset index 给数据加一列索引，让时间成为一个真正的列，而不是索引列，并更新数据
        df_ohlc.reset_index(inplace=True)
        # format date into matplot style 将时间变为数字形式，可以画图
        df_ohlc['Date'] = df_ohlc.Date.map(mdates.date2num)

        # plotting
        # in order to re-use the function individually
        # 因为一开始的默认值是[5,10,20,30]，所以如果用户重新写入moving_windows的值，则需要判断一下
        # 这一块就是可以完善和更新的地方，你可以加一些提示用户输入moving_windows的代码，这样就可以
        # 抛弃默认值，随心所欲
        if type(moving_windows) == list:
            for moving_window in moving_windows:
                # 求收盘价的滚动平均值
                df_ohlc['MA%d' % (moving_window)] = df_ohlc['close'].rolling(moving_window).mean()
            plt.figure(figsize=(15,8))
            # 画上面的子图，2grid第一个参数(10,1),意思是10行1列的意思，(0,0)是从原点开始画图，第一个子图跨度是7行1列
            ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
            # 第2格子图是从(8,0)开始画图，跨度是2行一列
            ax2 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1)
            # 第一个子图的横坐标是时间
            ax1.xaxis_date()
            ax1.set_ylabel('Price')
            # 这个就是画K线图的一个函数，用的就是df_ohlc的所有值
            candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
            # 设定好画图的基本元素以后，再针对每一个moving_window，画图
            for moving_window in moving_windows:
                ax1.plot(df_ohlc.Date, df_ohlc['MA%d' % (moving_window)], label='MA%d' % (moving_window))
            # 针对交易量，画柱状图
            ax2.bar(df_ohlc.Date, df_volume.values, width=1)
            # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
            ax2.set_ylabel('Volume')
            plt.suptitle(title)
            ax1.legend()
            plt.show()
        # 如果moving_windows只是一个数字，就不用不用遍历列表，直接取用就好，后面的代码是一样的，
        # 只是没有了遍历的过程，因为只需要画一条线
        elif type(moving_windows) == int:
            df_ohlc['MA%d' % (moving_windows)] = df_ohlc['close'].rolling(moving_windows).mean()
            plt.figure(figsize=(15,8))
            ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
            ax2 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1)
            ax1.xaxis_date()
            ax1.set_ylabel('Price')
            candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
            ax1.plot(df_ohlc.Date, df_ohlc['MA%d' % (moving_windows)], label='MA%d' % (moving_windows))
            ax2.bar(df_ohlc.Date, df_volume.values, width=1)
            ax2.set_ylabel('Volume')
            plt.suptitle(title)
            ax1.legend()
            plt.show()
        # 如果moving_windows是空值，就默认是一天，直接画图就好了，也不用求moving_average
        elif moving_windows is None:
            plt.figure(figsize=(15,8))
            ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
            ax2 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1)
            ax1.xaxis_date()
            ax1.set_ylabel('Price')
            candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
            ax2.bar(df_ohlc.Date, df_volume.values, width=1)
            ax2.set_ylabel('Volume')
            plt.suptitle(title)
            ax1.legend()
            plt.show()

    def plot_Kline_EMA(self, company_symbol=None, start_date=None, end_date=None, moving_windows=[5,10,20,30]):
        """ TODO - Kline with Expected weighted average, call function from utils.py"""
        print("Here are existed companies: ")
        _ = read_exist_companylist(display=True)

        if company_symbol is None:
            company_symbol = input("Please enter the symbols your selected company: ")

            if company_symbol not in self.company_names_list:
                print('%s is not in the existing company list, before you analyse it you need to download its data firstly...' % (company_symbol))
                company_symbol = company_symbol.upper()
                start_date = input('Please enter a start time, as format(yyyy-mm-dd): ')
                end_date = input('Please enter a end time, as format(yyyy-mm-dd): ')
                df = self.get_data_online(company_symbol=company_symbol, start_date=start_date, end_date=end_date)
            else:
                df = read_exist_data(company_symbol)
        else:
            if company_symbol not in self.company_names_list:
                print('%s is not in the existing company list, before you analyse it you need to download its data firstly...' % (company_symbol))
                company_symbol = company_symbol.upper()
                start_date = input('Please enter a start time, as format(yyyy-mm-dd): ')
                end_date = input('Please enter a end time, as format(yyyy-mm-dd): ')
                df = self.get_data_online(company_symbol=company_symbol, start_date=start_date, end_date=end_date)
            else:
                df = read_exist_data(company_symbol)

        print('You select %s to plot!' % (company_symbol))

        is_resample = input('\ncurrent data is daily chart, do you want to resample to generate a lower frequency K-lines, , format as Y/N:')
        if is_resample.lower() == 'y' or is_resample.lower() == 'yes':
            days = input("enter a time, format as 10D: ")
        else:
            days = '1D'

        # Set title
        title = '%s - %s-%say K-line' % (company_symbol, days[:-1], days[-1])

        if start_date is not None and end_date is not None:
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        elif start_date is not None and end_date is None:
            df = df[df.index >= pd.to_datetime(start_date)]
        elif start_date is None and end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        # Processing data
        # adjust OHL data according to "adj close"
        adj_factor = df['Adj Close'] / df['Close']
        df['Open'] = df['Open'] * adj_factor
        df['High'] = df['High'] * adj_factor
        df['Low'] = df['Low'] * adj_factor

        if days == '1D':
            df_ohlc = df[['Open','High','Low','Adj Close']]
            df_ohlc.columns = ['open', 'high', 'low', 'close']
            df_volume = df['Volume']
            pass
        else:
            # resampling
            df_ohlc = pd.DataFrame(df['Open'].resample(days).first(), columns=['Open'])
            df_ohlc['high'] = df['High'].resample(days).max()
            df_ohlc['low'] = df['Low'].resample(days).min()
            df_ohlc['close'] = df['Adj Close'].resample(days).last()
            df_ohlc.columns = ['open','high','low','close']
            df_volume = df['Volume'].resample(days).sum()
            # Drop nan
            df_ohlc.dropna(inplace=True)
            df_volume.dropna(inplace=True)

        # reset index 
        df_ohlc.reset_index(inplace=True)
        # format date into matplot style
        df_ohlc['Date'] = df_ohlc.Date.map(mdates.date2num)

        # plotting
        # in order to re-use the function individually
        # 因为默认值是一个列表，所以这里执行这一行代码
        if type(moving_windows) == list:
            plt.figure(figsize=(15, 8))
            ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
            ax2 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1)
            ax1.xaxis_date()
            ax1.set_ylabel('Price')
            # 这就是一个画K线图的函数，固定表达
            candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
            for moving_window in moving_windows:
                # exp_moving_average就是一个求指数滚动平均值的函数，以收盘价作为基准
                df_ohlc['EMA%d' % (moving_window)] = exp_moving_average(df_ohlc['close'], moving_window)
                ax1.plot(df_ohlc.Date, df_ohlc['EMA%d' % (moving_window)], label='EMA%d' % (moving_window))
            # 画第二个子图，交易量图
            ax2.bar(df_ohlc.Date, df_volume.values, width=1)
            # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
            ax2.set_ylabel('Volume')
            plt.suptitle(title)
            ax1.legend()
            plt.show()
        elif type(moving_windows) == int:
            plt.figure(figsize=(15, 8))
            ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
            ax2 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1)
            ax1.xaxis_date()
            ax1.set_ylabel('Price')
            candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
            df_ohlc['EMA%d' % (moving_windows)] = exp_moving_average(df_ohlc['close'], moving_windows)
            ax1.plot(df_ohlc.Date, df_ohlc['EMA%d' % (moving_windows)], label='EMA%d' % (moving_windows))
            ax2.bar(df_ohlc.Date, df_volume.values, width=1)
            # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
            ax2.set_ylabel('Volume')
            plt.suptitle(title)
            ax1.legend()
            plt.show()
        elif moving_windows is None:
            plt.figure(figsize=(15, 8))
            ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=7, colspan=1)
            ax2 = plt.subplot2grid((10, 1), (8, 0), rowspan=2, colspan=1, sharex=ax1)
            ax1.xaxis_date()
            ax1.set_ylabel('Price')
            candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
            ax2.bar(df_ohlc.Date, df_volume.values, width=1)
            # ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
            ax2.set_ylabel('Volume')
            plt.suptitle(title)
            ax1.legend()
            plt.show()
    # this part is to define the non-linear regression model, training the modeling period
    def model_training_nonlinear(self, x, y, model=None, **kwargs):  #  **kwargs传入字典表参数，后面会用到
        # TODO - training non-linear model
        '''Random forest regression model, more information can be found from http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'''
        # users can also define their own models to suit their requirements, more information can be found : http://scikit-learn.org/
        if model is None:
            # 如果不输入model，则自动调用这个回归函数
            clf = RandomForestRegressor(**kwargs)
        else:
            clf = model(**kwargs)

        # 训练模型
        clf.fit(x, y)
        print('Non-linear model training done!')
        return clf

    # nonlinear model - predict
    def model_prediction_nonlinear(self, x, model):
        """ TODO - nonlinear model - prediction """
        return model.predict(x)

    def model_training_linear(self, x, y, model=None, **kwargs):
        # TODO - linear model training
        """linear model，
        refer to：http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression"""
        # Can also customise training model here, other model see: http://scikit-learn.org/
        if model is None:
            clf = LinearRegression(**kwargs)
        else:
            clf = model(**kwargs)

        clf.fit(x, y)
        print('linear model training done!')
        return clf

    # prediction using linear regression model
    def model_prediction_linear(self, x, model):
        """ TODO - linear model prediction """
        return model.predict(x)


    def model_prediction_visualization(self, train_values, test_values, title=None):
        """ TODO - plotting，divide historical data into in-sample as training data (4/5),
        and out-of-sample prediciton (1/5)，with the true value"""
        # 设置一条线，区分训练数据和测试数据
        train_test_split_line = train_values.index[-1]
        # 把训练数据和测试数据单纯的拼接到一起，上下拼接
        concat_data = pd.concat([train_values, test_values], axis=0)
        # concat_data['actual_values'][-1] = 0
        plt.figure(figsize=(15,8))
        # 这里画图的话，以为两个数据是上下拼接到一起的，所以可以同时间段画图
        plt.plot(concat_data.index, concat_data['actual_values'], 'r-', label='actual values')
        plt.plot(concat_data.index, concat_data['fitted_values'], 'g-.', label='fitted values')
        ymin = min(concat_data['actual_values'].min(), concat_data['fitted_values'].min())
        ymax = max(concat_data['actual_values'].max(), concat_data['fitted_values'].max())
        plt.vlines(train_test_split_line,
                   ymin=ymin,
                   ymax=ymax,
                   linestyles='-.')
        plt.text(concat_data.index[0] + pd.Timedelta(days=20),
                 0.5*(ymin + ymax),
                 'In-sample')
        plt.text(train_test_split_line + pd.Timedelta(days=5),
                 0.5*(ymin + ymax),
                 'Out-of-sample')
        plt.legend()
        if title is not None:
            plt.title(title)
        plt.show()
        pass

    # 提示在下面子类具体写到这个函数并使用它，不写也没有关系
    def do_analysis(self):
        # TODO the function analysis
        # define this function first in base class
        pass




# TODO - # inherit base class Financial_Analysis_Basis，
# write the function do_analysis, which was defined before
class My_Analysis(Financial_Analysis_Basis):
    def process_data(self, data, N=5):
        """manage and analysis data for building model"""
        # 为了后面建模使用
        adj_close = data['close']
        shifted_data_list = []
        for i in range(N):
            shifted_data_list.append(data.shift(i))  # 往下调一个数据，然后上面设为NAN
        # 拼接数据，作为特征值，这个可以说就是问的，机器学习的知识，我们也不是特别懂
        # 横着拼接
        processed_data = pd.concat(shifted_data_list, axis=1)
        # drop NAN
        processed_data.dropna(inplace=True)
        # set study target, adj_close of the next day
        processed_data['targets'] = adj_close.shift(-1)
        return processed_data

    def do_analysis(self):
        # 开始做基本的分析
        print("Here are existed companies: ")
        # display=True，意味着调用函数以后，要展示出来
        _ = read_exist_companylist(display=True)

        # count the times of unusual occurs
        exception_times = 0
        # the maximum of the time that allow unusual occurs
        max_exception_times = 10

        while True:
            try:
                # to get the data
                print('\nPart 1 - Getting data!')
                company_symbol = input("enter a company's symbol: ")
                company_symbol = company_symbol.upper()

                start_date = input('enter a start time, as format(yyyy-mm-dd): ')
                end_date = input('enter the end time, as format(yyyy-mm-dd): ')
                # to see if company_symbol is in the company list
                if company_symbol in self.company_names_list:
                    # if company_symbol is in the company list, then download the exist data saved previously
                    download_data = True  # decide to download data
                    # 调用read函数，读取缓存数据
                    buffer_data = read_exist_data(company_symbol)
                    # 将选取的缓存数据的时间范围放到一个列表里面
                    data_range = [buffer_data.index.min().strftime('%Y-%m-%d'),
                                  buffer_data.index.max().strftime('%Y-%m-%d')]
                    # if the start date is equal or larger than the start date of the exist saved data
                    #and the end date equal or smaller than the end date of the exist data
                    # then the saved data already coverd the date requested by the user
                    #so there is no need to download more data from online
                    if start_date >= data_range[0] and end_date <= data_range[1]:
                        used_data = buffer_data[(buffer_data.index >= start_date) &
                                                (buffer_data.index <= end_date)]
                        # 说明选择的时间段小于缓存的数据，则无需再下载数据
                        download_data = False
                    else:
                        # just copy the data
                        # 如果选择的时间段大于缓存数据，我们就需要拼接新数据进来
                        # 首先，我们需要先copy一下缓存的数据，为了一会拼接新的数据
                        used_data = buffer_data.copy()

                    # 这里如果无需缓存，download_data=False，则不进行下列语句
                    # 因为我们之前设定download_data=True，所以如果需要缓存，我们的download_data
                    # 就会等于True，然后执行下面的语句
                    if download_data:
                        # only downloads the part that not included in the exist part
                        #and then joint them together
                        # 首先拼接时间处在缓存数据前面的数据
                        new_data_part_i = self.get_data_online(company_symbol,
                                                               start_date=min(start_date, data_range[0]),
                                                               end_date=max(start_date, data_range[0]))
                        new_data_part_ii = self.get_data_online(company_symbol,
                                                                start_date=min(end_date, data_range[1]),
                                                                end_date=max(end_date, data_range[1]))
                        # put the data together, joint them and sort the data
                        used_data = pd.concat([buffer_data, new_data_part_i, new_data_part_ii], axis=0).sort_index()
                        # save new data and cover the old exist data
                        used_data.to_csv(os.path.join(data_dir, '{}.csv'.format(company_symbol)))

                    # 将used_data重新设定索引，让时间成为一个正常的列
                    used_data = used_data.reset_index()
                    # 去掉重复的列，以时间为基准
                    used_data = used_data.drop_duplicates(subset='Date')
                    # 重新将used_data的索引列设为Date
                    used_data = used_data.set_index('Date')
                else:
                    # 如果选取的公司根本不在company list里面，则直接调用get data函数下载数据即可
                    used_data = self.get_data_online(company_symbol, start_date=start_date, end_date=end_date)
                    write_company_name_list(company_symbol)
                    # 调用更新公司列表的函数，确保数据已更新
                    self._refresh_company_names_list()

                # this part is about statistics analysis and related graphs
                # statistics analysis
                print('\nPart 2 - Descriptive analysis!')
                self.descriptive_analytics(company_symbol=company_symbol)

                # draw k-line graph
                print('\nPart 3 - Visualise analysis!')
                print('Here are some further visualise analysis,')
                print('1. K-line with EMA \n2. K-line with MA \n3. Quit')
                ma_or_ema = input("Please enter your choose option:")
                while ma_or_ema != "3":
                    if ma_or_ema == '1':
                        # 调用基类已经写好的画图函数
                        self.plot_Kline_EMA(company_symbol=company_symbol,
                                            start_date=start_date,
                                            end_date=end_date,
                                            moving_windows=[5,10,20,30])
                    elif ma_or_ema == '2':
                        self.plot_Kline_MA(company_symbol=company_symbol,
                                           start_date=start_date,
                                           end_date=end_date,
                                           moving_windows=[5, 10, 20, 30])
                    else:
                        print("Wrong choice, please try again.")
                    print('1. K-line with EMA \n2. K-line with MA \n3. Quit')
                    ma_or_ema = input("Please enter your choose option:")
                # TODO - prediction part
                # this part is about the prediction
                # in order to predict the close price for the next day,
                #use the past N days' data includes adj open，adj high，adj low and adj close
                # adjust OHL
                adj_factor = used_data['Adj Close'] /  used_data['Close']
                used_data['Open'] = used_data['Open'] * adj_factor
                used_data['High'] = used_data['High'] * adj_factor
                used_data['Low'] = used_data['Low'] * adj_factor

                # TODO - whether to resample before model training and prediction
                print('\nPart 4 - Model training and prediction!')
                is_resample = input('current data is daily, do you want to resample to predict a lower frequency data, format as Y/N:')
                if is_resample.lower() == 'y' or is_resample.lower() == 'yes':
                    days = input("enter a time, format as 10D: ")
                else:
                    days = '1D'

                if days == '1D':
                    df_ohlc = used_data[['Open','High','Low','Adj Close']]
                    df_ohlc.columns = ['open', 'high', 'low', 'close']
                else:
                    # resampling
                    df_ohlc = pd.DataFrame(used_data['Open'].resample(days).first(), columns=['Open'])
                    df_ohlc['high'] = used_data['High'].resample(days).max()
                    df_ohlc['low'] = used_data['Low'].resample(days).min()
                    df_ohlc['close'] = used_data['Adj Close'].resample(days).last()
                    df_ohlc.columns = ['open','high','low','close']
                    # Drop nan
                    df_ohlc.dropna(inplace=True)

                # choose a period of time as training data
                #then the rest of time is the test data
                # used to support the prediction
                training_start_date = input('enter your desired start time for training, as format(yyyy-mm-dd): ')
                training_end_date = input('enter your desired end time for training, as format(yyyy-mm-dd): ')
                linear_or_nonlinear = input('enter to indicate if to use linear or not, as format(Y/N):')
                # Process Data
                # 传入数据，然后做5次特征值数据集
                processed_used_data = self.process_data(df_ohlc[['open', 'high', 'low', 'close']])

                # Deriv e training_data and test_data
                # 训练数据集是需要选择时间段的
                training_data = processed_used_data[(processed_used_data.index >= pd.to_datetime(training_start_date)) &
                                                    (processed_used_data.index <= pd.to_datetime(training_end_date))]
                # 测试数据集就是除去训练数据以外的剩下的数据，所以是大于end_date的index
                test_data = processed_used_data[processed_used_data.index > pd.to_datetime(training_end_date)]

                # 这一块还是机器学习的部分，原理不懂
                # [:, [:-1]] 意思是去掉最后一列，然后选择所有的行，作为训练特征
                training_features = training_data.iloc[:, :-1]
                # 选择最后一列作为目标
                training_targets = training_data.iloc[:, -1]
                test_features = test_data.iloc[:, :-1]
                test_targets = test_data.iloc[:, -1]

                # TODO - model training
                if linear_or_nonlinear.lower() == 'n' or linear_or_nonlinear.lower() == 'no':
                    model = self.model_training_nonlinear(x=training_features.values, y=training_targets.values,
                                                          **nonlinear_model_config)
                    train_model_preds = self.model_prediction_nonlinear(x=training_features.values, model=model)
                    test_model_preds = self.model_prediction_nonlinear(x=test_features.values, model=model)
                    title = '%s Non-linear Model Prediction' % (company_symbol)
                else:
                    model = self.model_training_linear(x=training_features.values, y=training_targets.values,
                                                       **linear_model_config)
                    train_model_preds = self.model_prediction_linear(x=training_features.values, model=model)
                    test_model_preds = self.model_prediction_linear(x=test_features.values, model=model)
                    title = '%s Linear Model Prediction' % (company_symbol)

                # TODO - plot figures
                training_data['actual_values'] = training_targets
                training_data['fitted_values'] = train_model_preds
                test_data['actual_values'] = test_targets
                test_data['fitted_values'] = test_model_preds

                self.model_prediction_visualization(training_data[['actual_values', 'fitted_values']], test_data[['actual_values', 'fitted_values']], title=title)


                # TODO - get predicted price
                # ask user to enter one day and provide the predicted price, if it is historic data, then print the true value.
                # if it is the next day's value, print null true value.
                print('\nPart 5 - Model prediction for given days!')
                part_5_exception_times = 0
                max_part_5_exception_times = 5
                # turn data into original data
                test_data.index = test_data.index.shift(int(days[:-1]), freq='D')

                while True:
                    print('There are some dates you can select:')
                    # 只要给出测试集数据的index就好，因为训练集做预测是没有意义的，一定是对的
                    # pprint就是一个更好的输出方式，让dataframe的格式更加标准，输出的是一个时间列表
                    pprint(list(test_data.index.strftime('%Y-%m-%d')))
                    selective_date = input('Please enter a date you want to check the predicted values, format as(yyyy-mm-dd or exit):')
                    try:
                        if selective_date != 'exit':
                            # TODO: print out predicted value for one day, only provide test data's predicted value and true value.
                            # 将输入的时间格式转化为标准的时间格式
                            # 输入可以为1991-01-01，19910101，1991-1-1
                            # 最终都会转化为：Timestamp('1991-01-01 00:00:00')
                            selective_date_dn = pd.to_datetime(selective_date)
                            if selective_date_dn not in test_data.index:
                                print('Your selected date is unavailable, please re-enter...')
                                # 这里要加一个continue，一旦不在列表里面，则自动跳出if语句，进行下一次循环
                                # 如果不加的话，如果用户只是输入不存在的日期，也会算作exception
                                # 但其实我们的用意是，只有用户在输入完全错误的东西时，才会报错。
                                continue
                            print('{} predicted value is {}, actual value is {}.'.format(selective_date,
                                                                                           test_data['fitted_values'].loc[selective_date],
                                                                                           test_data['actual_values'].loc[selective_date]))
                            _ = input('Press any to continue...\n')
                        else:
                            # TODO - quit
                            print('All done! Welcome back!')
                            break
                    # 设定输错次数，如果大于5次，则自动退出
                    except Exception as e:
                        part_5_exception_times += 1
                        if part_5_exception_times >= max_part_5_exception_times:
                            print('Too many exceptions, then skip out!')
                            break
                        print('There is a exception, info: {}, please re-enter...'.format(e))

                break
            # 这里是如果输错次数大于10次，自动退出
            except Exception as e:
                exception_times += 1
                if exception_times >= max_exception_times:
                    print('Too many exceptions, then program terminated!')
                    break
                print('There is a exception, info: {}. Please Try again...'.format(e))
