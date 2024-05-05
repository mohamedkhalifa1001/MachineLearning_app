import streamlit as st
import tensorflow  as  tf
from tensorflow import keras
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from cleaning_drybeans import cleaning_drybeans
from sklearn.tree import DecisionTreeClassifier
import joblib


st.title("Dry Beans Classification")
#hyperparamaters of nn 
st.sidebar.subheader("Neural network hyperparameters")
num_epochs=st.sidebar.slider("Enter number of Epochs :",1,20)
hidden_activation=st.sidebar.text_input("Enter activation function in Hidden layers :")
output_activation=st.sidebar.text_input("Enter activation function in output layer :")

#hyperparameters of decision tree
st.sidebar.subheader("Decision Tree Hyperparameters")
max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=800, value=5)
min_samples_split = st.sidebar.slider("Min Samples Split", min_value=2, max_value=500, value=2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", min_value=1, max_value=200, value=1)

#reading the dataset as dataframe
dry = pd.read_csv("datasets\Dry_Bean_Dataset.csv")

#let's clean the data and remove the outliers 
data=cleaning_drybeans(dry)


# split data to features and target 
X = data.iloc[:, :-1].values
y = data.iloc[:, -1:].values

# split data to train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#scale the data
scaler = StandardScaler(copy=True, with_std=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


if st.button("Train Neural Network model"):
    # Set random seed for reproducibility
    np.random.seed(0)
    tf.random.set_seed(0)
    model = keras.Sequential([
    keras.layers.Dense(128, activation=hidden_activation, input_shape=(X_train.shape[1],)),
    keras.layers.Dense(110, activation=hidden_activation),
    keras.layers.Dense(110, activation=hidden_activation),
    keras.layers.Dense(7, activation=output_activation)#  # Number of unique classes
])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'sparse_categorical_crossentropy' since your target is not one-hot encoded
              metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(X_test,y_test))
    # Save the model
    model.save("dry_beans.h5")
    st.write('The NN model is trained successfully')

if st.button('Evaluate Neural Network'):
    # Load the model
    model = keras.models.load_model("dry_beans.h5")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    st.write("The test loss :",test_loss)
    st.write("The test accuracy",test_accuracy)



if st.button("Train Desicion tree model"):
    tree_clf = DecisionTreeClassifier(
        max_depth=max_depth,  # Maximum depth of the tree
        min_samples_split=min_samples_split,  # Minimum number of samples required to split an internal node
        min_samples_leaf=min_samples_leaf,  # Minimum number of samples required to be a leaf node
        random_state=42  
    )
    # Train the classifier on the training data
    tree_clf.fit(X_train, y_train)
   

    # Save the Decision Tree model
    joblib.dump(tree_clf, "decision_tree.pkl")
    st.write("The tree classifier is trained Successfully ")



if st.button("Evaluate Decision tree"):
    try:
          tree_clf = joblib.load("decision_tree.pkl")
          y_pred = tree_clf.predict(X_test)
          st.write("train score: ",tree_clf.score(X_train, y_train))
          st.write("test score: ",tree_clf.score(X_test, y_test))
          accuracy = accuracy_score(y_test, y_pred)
          report = classification_report(y_test, y_pred)
          st.write("Decision Tree Classifier Accuracy:", accuracy)
          st.write("Classification Report:\n", report)


    except FileNotFoundError:
        st.write('please train the model first')

st.subheader("Make Classification on feature values")
input_values=st.text_input("Enter feature values :")

if st.button("Make Classification"):
    try:
        # Load the trained models
        nn_model = keras.models.load_model("dry_beans.h5")
        dt_model = joblib.load("decision_tree.pkl")
        # Load the test data
        test_data = pd.read_csv("datasets\Dry_test_data.csv")  
   

        # Preprocess the input test data
        input_values = input_values.strip()  # Remove leading and trailing whitespace
        input_values = input_values.replace('[', '').replace(']', '')  # Remove square brackets if present
        input_values = [float(value.strip()) for value in input_values.split(',')]  # Convert to float and remove leading/trailing whitespace
        input_values = np.array(input_values).reshape(1, -1)  # Reshape to match the input shape
        input_values_scaled = scaler.transform(input_values) 

        # the target labels 
        classes=['SEKER','BARBUNYA','BOMBAY','CALI','HOROZ','SIRA','DERMASON']

        # Predict using Neural Network model
        nn_prediction = nn_model.predict(input_values_scaled)
        nn_predicted_class_index = np.argmax(nn_prediction)
        nn_predicted_class = classes[nn_predicted_class_index]
        # Predict using Decision Tree model
        dt_prediction = dt_model.predict(input_values_scaled)
        dt_predicted_class_index = dt_prediction[0]
        dt_predicted_class = classes[dt_predicted_class_index]

        # Find the index of the row in the test data that matches the input values
        test_data_index = test_data.iloc[:, :-1].eq(input_values.flatten()).all(axis=1).idxmax()

        # Extract the actual class label for the matching row
        actual_class = test_data.iloc[test_data_index, -1]

        # Calculate accuracy
        nn_test_accuracy = 1 if nn_predicted_class == actual_class else 0
        dt_test_accuracy = 1 if dt_predicted_class == actual_class else 0

        # Display the results
        st.write("Predicted Class (Neural Network):", nn_predicted_class)
        st.write("Predicted Class (Decision Tree):", dt_predicted_class)
        st.write("Actual Class:", actual_class)
        st.write("Neural Network Model Accuracy:", nn_test_accuracy)
        st.write("Decision Tree Model Accuracy:", dt_test_accuracy)

        
    except FileNotFoundError:
        st.write("Please make sure you have uploaded the test data file and trained both Neural Network and Decision Tree models.")
