import numpy as np
import pandas as pd

def cleaning_drybeans(dry):
    dry = dry.drop_duplicates()
    def handle_outliers_iqr(dataframe, column):
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = dataframe[(dataframe[column] < lower) | (dataframe[column] > upper)]
        dataframe.loc[(dataframe[col]>upper,col)]=upper
        dataframe.loc[(dataframe[col]<lower,col)]=lower 
        # np.where(condition, if True excute this, NO excute this)
        # dataframe[column] = np.where((dataframe[column] < lower) | (dataframe[column] > upper), dataframe[column].median(), dataframe[column])
        return dataframe
    

    for col in dry.select_dtypes(include=["number"]).columns :
        Dry_beans = handle_outliers_iqr(dry, col)
    # convert object classes to numircal value
    dry.replace({"Class":{"SEKER":0, "BARBUNYA":1, "BOMBAY":2, "CALI":3, "HOROZ":4, "SIRA":5, "DERMASON":6}}, inplace=True)
    newClasses = dry["Class"].unique()

    return Dry_beans
    