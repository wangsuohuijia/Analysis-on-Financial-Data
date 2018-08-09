"""
    authors:
             Suohuijia Wang,   17200170
"""

# -*- coding: UTF-8 -*-
from project_class_def import My_Analysis

if __name__ == '__main__':
    # create objects
    my_analysis_worker = My_Analysis()
    # do analysis
    # it is divided into 4 parts: get data, descriptive analysis, plot figures, prediction.
    my_analysis_worker.do_analysis()
