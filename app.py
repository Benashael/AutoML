import streamlit as st
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Define a function to load your dataset
@st.cache
def load_data():
    # Load your dataset here using Pandas
    data = pd.read_csv("your_dataset.csv")
    return data

# Create Streamlit pages
page = st.sidebar.selectbox("Select a Page", ["Data Cleaning", "Data Encoding", "Data Visualization", "ML Model Selection", "AutoML for Regression", "AutoML for Classification", "AutoML for Clustering"])

if page == "Data Cleaning":
    st.title("Data Cleaning App Page")
    data = load_data()
    st.write("Dataset:")
    st.write(data)
    
    if st.checkbox("Show Summary Stats"):
        st.write("Summary Statistics:")
        st.write(data.describe())
    
    st.write("Missing Values:")
    st.write(data.isnull().sum())
    
    if st.checkbox("Handle Missing Values"):
        st.subheader("Handle Missing Values")
        operation = st.radio("Select Operation", ("Drop Rows", "Impute with Mean", "Impute with Median", "Custom Value"))
        if operation == "Drop Rows":
            data = data.dropna()
        elif operation == "Impute with Mean":
            data = data.fillna(data.mean())
        elif operation == "Impute with Median":
            data = data.fillna(data.median())
        else:
            custom_value = st.number_input("Enter Custom Value", value=0)
            data = data.fillna(custom_value)
    
    st.write("Data After Handling Missing Values:")
    st.write(data)

elif page == "Data Encoding":
    st.title("Data Encoding App Page")
    data = load_data()
    st.write("Dataset:")
    st.write(data)
    
    st.subheader("Encode Categorical Variables")
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        st.write(f"Encoding '{col}' column")
        encoding_option = st.radio("Select Encoding Method", ("Label Encoding", "One-Hot Encoding"))
        if encoding_option == "Label Encoding":
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
        else:
            data = pd.get_dummies(data, columns=[col], prefix=[col])
    
    st.write("Data After Encoding Categorical Variables:")
    st.write(data)

elif page == "Data Visualization":
    st.title("Data Visualization App Page")
    data = load_data()
    st.write("Dataset:")
    st.write(data)
    
    st.subheader("Data Visualization")
    st.write("Select Columns for Visualization:")
    columns_to_visualize = st.multiselect("Select Columns", data.columns)
    if columns_to_visualize:
        st.line_chart(data[columns_to_visualize])
        st.bar_chart(data[columns_to_visualize])
        st.area_chart(data[columns_to_visualize])

elif page == "ML Model Selection":
    st.title("ML Model Selection App Page")
    data = load_data()
    st.write("Dataset:")
    st.write(data)
    
    st.subheader("Select Problem Type")
    problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])
    
    if problem_type == "Classification":
        st.write("Example: Recommend Classification Models using Scikit-learn")
        X = data.drop("target_column", axis=1)
        y = data["target_column"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = [RandomForestClassifier()]
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f"Model: {type(model).__name__}")
            st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

    elif problem_type == "Regression":
        st.write("Example: Recommend Regression Models using Scikit-learn")
        X = data.drop("target_column", axis=1)
        y = data["target_column"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = [RandomForestRegressor()]
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write(f"Model: {type(model).__name__}")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

elif page == "AutoML for Regression":
    st.title("AutoML for Regression App Page")
    data = load_data()
    st.write("Dataset:")
    st.write(data)
    
    st.subheader("AutoML for Regression")
    target_variable = st.selectbox("Select Target Variable", data.columns)
    
    test_size = st.slider("Select Test Size (Fraction)", 0.1, 0.5, 0.2, 0.01)
    random_state = st.slider("Select Random State", 1, 100, 42, 1)
    
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    st.write("Regression Models:")
    regression_models = [RandomForestRegressor(), LinearRegression(), Ridge(), Lasso()]
    
    for model in regression_models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"Model: {type(model).__name__}")
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

elif page == "AutoML for Classification":
    st.title("AutoML for Classification App Page")
    data = load_data()
    st.write("Dataset:")
    st.write(data)
    
    st.subheader("AutoML for Classification")
    target_variable = st.selectbox("Select Target Variable", data.columns)
    
    test_size = st.slider("Select Test Size (Fraction)", 0.1, 0.5, 0.2, 0.01)
    random_state = st.slider("Select Random State", 1, 100, 42, 1)
    
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    st.write("Classification Models:")
    classification_models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]
    
    for model in classification_models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write(f"Model: {type(model).__name__}")
        st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

elif page == "AutoML for Clustering":
    st.title("AutoML for Clustering App Page")
    data = load_data()
    st.write("Dataset:")
    st.write(data)

    st.subheader("AutoML for Clustering")
    num_clusters = st.slider("Select the number of clusters:", 2, 10)
    X = data.dropna()

    # Check if there are enough data points for clustering
    if X.shape[0] < num_clusters:
        st.error("Not enough data points for the selected number of clusters.")
    else:
        # Perform one-hot encoding for categorical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_features:
            X_encoded = pd.get_dummies(X, columns=categorical_features)
        else:
            X_encoded = X

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        X_encoded['Cluster'] = kmeans.fit_predict(X_encoded)
        st.write(f"Performed K-Means clustering with {num_clusters} clusters.")
        st.write(X_encoded)

# Handle errors and optimize performance
try:
    if data is not None:
        st.write("Data Loaded Successfully")
    else:
        st.write("Data Not Loaded. Please upload a dataset.")
except Exception as e:
    st.error(f"An error occurred: {e}")

