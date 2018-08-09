# -*- coding: UTF-8 -*-
import os
from dir_info import buffer_dir, data_dir
import numpy as np
import pandas as pd

# TODO - general-purpose function

def write_company_name_list(company_symbol):
    """write every company symbol to a company list"""
    with open(os.path.join(buffer_dir, "companylist.txt"), "a") as f:
        f.write(str(company_symbol))
        f.write('\n')

def read_exist_companylist(display=False):
    """Display company symbols from 'companylist.txt'"""
    company_names_list = []
    for line in open(os.path.join(buffer_dir, "companylist.txt")):
        # print(line, end = "\n")
        # print(line)
        if line != '\n':
            company_names_list.append(line[:-1])
            if display:
                print(line[:-1])

    return company_names_list

def exp_moving_average(values, window):
    """expected weighted moving average """
    """WMA = (P1 * 5) + (P2 * 4) + (P3 * 3) + (P4 * 2) + (P5 * 1) / (5 + 4+ 3 + 2 + 1)"""
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = None
    return a

def read_exist_data(company_symbol):
    """Read data which has already been downloaded."""
    exist_data = pd.read_csv(os.path.join(data_dir, "{}.csv".format(company_symbol)),
                             parse_dates=True, index_col=0)
    return exist_data
