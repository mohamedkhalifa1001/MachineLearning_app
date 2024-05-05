import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
import joblib
from cleaning_houses import cleaning_houses


st.title("House pricing prediction using SVM")
#Hyper parameters of the support vector machine 
st.sidebar.subheader(" Hyperparameters")
kernel = st.sidebar.selectbox('Kernel', ['poly', 'linear', 'rbf', 'sigmoid'])
C=st.sidebar.text_input("Regulization value :")
C=float(C)
#reading the dataset as a Dataframe
houses = pd.read_csv("datasets/train (1).csv")

#let's clean our data
houses=cleaning_houses(houses)


 # split data to features and target 
X = houses.iloc[:, :-1].values
y = houses.iloc[:, -1:].values

 # split data to train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#scale the data
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if st.button("Train the model"):

    
    svr = SVR(kernel=kernel, C=C)
    svr.fit(X_train, y_train.ravel())  # Flatten y_train to avoid the ValueError

    y_train_pred = svr.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)


    y_test_pred = svr.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)


    # Save the model
    joblib.dump(svr, 'svm_model.pkl')

    # Save metrics
    metrics_dict = {'train_mae': train_mae, 'train_mse': train_mse, 'train_rmse': train_rmse,
                    'test_mae': test_mae, 'test_mse': test_mse, 'test_rmse': test_rmse}
    joblib.dump(metrics_dict, 'metrics.pkl')
    st.write("The model is trained successfully")

# Display evaluation metrics
if st.button("Evaluate the model"):
    try:
        svr = joblib.load('svm_model.pkl')
        metrics_dict = joblib.load('metrics.pkl')
        # score for train 
        st.write("train score: ",svr.score(X_train, y_train))
        # score for test 
        st.write(f"test score: ",svr.score(X_test, y_test))

        st.write("Train Metrics:")
        st.write("Mean Absolute Error: ",metrics_dict['train_mae'])
        st.write("Mean Squared Error: ",metrics_dict['train_mse'])
        st.write("Root Mean Squared Error: ",metrics_dict['train_rmse'])
       

        st.write("Test Metrics:")
        st.write("Mean Absolute Error: ",metrics_dict['test_mae'])
        st.write("Mean Squared Error: ",metrics_dict['test_mse'])
        st.write("Root Mean Squared Error: ",metrics_dict['test_rmse'])

    
      
    except FileNotFoundError:
        st.write("Please train the model first.")

# Add a text input field to take the features values 
input_values = st.text_input("Enter all feature values to make predections:")

if st.button("Predict"):
    try:
        # Load the trained model
        svr = joblib.load('svm_model.pkl')
        # Load the test data
        test_data = pd.read_csv('datasets/test_houses.csv')
        # Drop the 'Id' column if present
        if 'Id' in test_data.columns:
            test_data = test_data.drop('Id', axis=1)

        # Save the modified dataset
        test_data.to_csv('modified_test_houses.csv', index=False)

        # Preprocess the input
        input_values = input_values.strip()  # Remove leading and trailing whitespace
        input_values = input_values.replace('[', '').replace(']', '')  # Remove square brackets if present
        input_features = [float(value.strip()) for value in input_values.split(',')]
         # Scale the input features using StandardScaler
        scaler = StandardScaler()
        input_features_scaled = scaler.fit_transform([input_features])

        # Make prediction by the trained model
        prediction = svr.predict([input_features])[0]

        # Find the index of the row in the test data that matches the input values
        test_data_index = test_data.iloc[:, :-1].eq(input_features).all(axis=1).idxmax()

        # Extract the actual price for the matching row
        actual_price = test_data.iloc[test_data_index, -1]

        
        # Compute the Mean Squared Error
        mse = mean_squared_error([actual_price], [prediction])

        # Display the results
        st.write("Predicted House Price:", prediction)
        st.write("Actual House Price:", actual_price)
        st.write("Mean Squared Error:", mse)

    except FileNotFoundError:
        st.write("Please train the model first.")







