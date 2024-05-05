from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd



def cleaning_houses(houses):


    #Handling missing columns in all houses
    missing_data = houses.isnull().sum().sort_values(ascending=False)

    # percent missing_data in each column
    percent = (houses.isnull().sum()/houses.isnull().count()).sort_values(ascending=False)

    # selecting only the columns with missing data 
    missing_columns = missing_data[missing_data > 0]

    houses.drop(columns="Id").select_dtypes(include=['number']).columns.shape[0]
    houses.drop(columns="Id").select_dtypes(include=['number']).columns

    # correlation matrix 
    corr = houses.drop(columns='Id').corr(numeric_only=True)
    correlation=corr["SalePrice"].apply(abs).sort_values(ascending=False).reset_index()



    # columns with missing data
    missing_data = pd.concat([missing_data, percent], axis=1, keys=['Total', 'Percent'])

    #Cleaning and preparing the data
    houses = houses.drop((missing_data[missing_data['Total'] > 81]).index,axis=1) # will remove 7 feature
    houses = houses.drop(houses.loc[houses['Electrical'].isnull()].index) # remove one row that contain NaN
    houses = houses.drop(correlation.iloc[21: , 0].values,axis=1)
    # find name of columns that has dtypes => number
    name_of_coll = houses.drop(columns="Id").select_dtypes(include=['number']).columns

    # fillna in each column using sample()
    for col in name_of_coll :
        nan_indices = houses[col].isnull() # find place(index) of NaN
        random_samples = houses[col].dropna().sample(n=nan_indices.sum(), replace=True) # sample of coll without NaN
        houses.loc[nan_indices, col] = random_samples.values


    col_has_numbers = houses.drop(columns="Id").select_dtypes(include=['number'])

    name_of_coll = houses.drop(columns="Id").select_dtypes(include=['object']).columns

    for col in name_of_coll :
        # using mode()
        mode_for_coll = houses[col].mode()[0]
        houses[col].fillna(mode_for_coll, inplace=True)
        

    col_has_numbers = houses.drop(columns="Id").select_dtypes(include=['object'])


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

    pd.DataFrame(columns=["id",'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V28',])


    for col in houses.drop(columns="Id").select_dtypes(include=["number"]).columns :
        houses = handle_outliers_iqr(houses, col)

    # feature encodign object (text) columns
    obj_col = houses.drop(columns="Id").select_dtypes(include=['object']).columns
    obj_col = pd.DataFrame(obj_col,columns=["text col"])

    # using labelEncoder to convert objects
    encoder = LabelEncoder()
    for col in obj_col.values.flatten():
        houses[col]= encoder.fit_transform(houses[col])

    # Drop the column id
    column_to_drop = 'Id'
    houses= houses.drop(column_to_drop, axis=1)




    return houses