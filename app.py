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
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report

st.set_page_config(page_title="AutoML Application", page_icon="🤖", layout="wide")

# Define a function to load your dataset
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Upload a dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    data = None

# App description
st.title("AutoML Application")
st.write("This application allows you to perform various AutoML tasks, including data cleaning, encoding, visualization, model selection, and more. You can upload your dataset and choose from a variety of machine learning tasks.")

# Create Streamlit pages
page = st.sidebar.selectbox("Select a Page", ["Data Cleaning", "Data Encoding", "Data Visualization", "ML Model Selection", "AutoML for Classification", "AutoML for Regression", "AutoML for Clustering", "Model Evaluation"])

if page == "Data Cleaning":
    st.title("Data Cleaning App Page")
    if data is not None:
        st.write("Dataset:")
        st.write(data)
    
        if st.checkbox("Show Summary Stats"):
            st.write("Summary Statistics:")
            st.write(data.describe())
    
        st.write("Missing Values:")
        st.write(data.isnull().sum())
    
        if st.checkbox("Handle Missing Values"):
            st.subheader("Handle Missing Values")
            operation = st.radio("Select Operation", ["Drop Rows", "Impute with Mean", "Impute with Median", "Custom Value"])
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
    else:
        st.warning("Please upload a dataset to continue.")

elif page == "Data Encoding":
    st.title("Data Encoding App Page")
    if data is not None:
        st.write("Dataset:")
        st.write(data)
        st.write("Dataset Shape:")
        st.write(data.shape)

        # Define the maximum allowed dataset size for data encoding
        max_rows_for_encoding = 10000
        max_columns_for_encoding = 100

        if data.shape[0] > max_rows_for_encoding or data.shape[1] > max_columns_for_encoding:
            st.warning(f"Note: The dataset size exceeds the maximum allowed for data encoding (max rows: {max_rows_for_encoding}, max columns: {max_columns_for_encoding}).")
        else:
            st.subheader("Encode Categorical Variables")
            categorical_cols = data.select_dtypes(include=["object"]).columns

            if not categorical_cols.empty:
                selected_cols = st.multiselect("Select Categorical Columns to Encode", categorical_cols)

                if not selected_cols:
                    st.warning("Please select one or more categorical columns to encode.")
                else:
                    for col in selected_cols:
                        encoding_option = st.radio(f"Select Encoding Method for '{col}'", ["Label Encoding", "One-Hot Encoding"])

                        if encoding_option == "Label Encoding":
                            le = LabelEncoder()
                            data[col] = le.fit_transform(data[col])
                        else:
                            data = pd.get_dummies(data, columns=[col], prefix=[col])
                    
                    st.write("Data After Encoding Categorical Variables:")
                    st.write(data)
            else:
                st.warning("No categorical columns found in the dataset for encoding.")
    else:
        st.warning("Please upload a dataset to continue.")

elif page == "Data Visualization":
    st.title("Data Visualization App Page")
    if data is not None:
        st.write("Dataset:")
        st.write(data)
    
        st.subheader("Data Visualization")
        st.write("Select Columns for Visualization:")
        columns_to_visualize = st.multiselect("Select Columns", data.columns)
        if columns_to_visualize:
            st.line_chart(data[columns_to_visualize])
            st.bar_chart(data[columns_to_visualize])
            st.area_chart(data[columns_to_visualize])
    else:
        st.warning("Please upload a dataset to continue.")
        
elif page == "ML Model Selection":
    st.title("ML Model Selection App Page")
    if data is not None:
        st.write("Dataset:")
        st.write(data)
        st.write("Dataset Shape:")
        st.write(data.shape)

        # Define the maximum allowed dataset size for ML model selection
        max_rows_for_ml_selection = 5000
        max_columns_for_ml_selection = 50

        if data.shape[0] > max_rows_for_ml_selection or data.shape[1] > max_columns_for_ml_selection:
            st.warning(f"Note: The dataset size exceeds the maximum allowed for ML model selection (max rows: {max_rows_for_ml_selection}, max columns: {max_columns_for_ml_selection}).")
        else:
            st.subheader("Select Problem Type")
            problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])

            if problem_type == "Classification":
                st.write("Example: Recommend Classification Models using Scikit-learn")
            else:
                st.write("Example: Recommend Regression Models using Scikit-learn")

            target_variable = st.selectbox("Select Target Variable", data.columns)

            X_columns = [col for col in data.columns if col != target_variable]
            selected_columns = st.multiselect("Select Features (X)", X_columns)

            if not selected_columns:
                st.warning("Please select one or more features (X) before running the ML model selection.")
            else:
                if problem_type == "Classification":
                    X = data[selected_columns]
                    y = data[target_variable]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    st.write("Classification Models:")
                    classification_models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]
                    for model in classification_models:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write(f"Model: {type(model).__name__}")
                        st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
                elif problem_type == "Regression":
                    X = data[selected_columns]
                    y = data[target_variable]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    st.write("Regression Models:")
                    regression_models = [RandomForestRegressor(), LinearRegression(), Ridge(), Lasso()]
                    for model in regression_models:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write(f"Model: {type(model).__name__}")
                        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    else:
        st.warning("Please upload a dataset to continue.")

elif page == "AutoML for Classification":
    st.title("AutoML for Classification App Page")
    if data is not None:
        st.write("Dataset:")
        st.write(data)
        st.write("Dataset Shape:")
        st.write(data.shape)

        st.subheader("AutoML for Classification")

        # Define the maximum allowed dataset size for classification
        max_rows_for_classification = 5000
        max_columns_for_classification = 50

        if data.shape[0] > max_rows_for_classification or data.shape[1] > max_columns_for_classification:
            st.error(f"Dataset size exceeds the maximum allowed for classification (max rows: {max_rows_for_classification}, max columns: {max_columns_for_classification}).")
        else:
            target_variable = st.selectbox("Select Target Variable (Y)", data.columns)
            st.write("Select X Variables:")
            X_variables = st.multiselect("Select Features (X)", [col for col in data.columns if col != target_variable])

            if not X_variables:
                st.warning("Please select one or more features (X) before running the AutoML for classification.")
            else:
                test_size = st.slider("Select Test Size (Fraction)", 0.1, 0.5, 0.2, 0.01)
                random_state = st.slider("Select Random State", 1, 100, 42, 1)

                X = data[X_variables]
                Y = data[target_variable]

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

                st.write("Classification Models:")
                classification_models = [RandomForestClassifier(), LogisticRegression(), DecisionTreeClassifier()]
                for model in classification_models:
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)
                    st.write(f"Model: {type(model).__name__}")
                    st.write(f"Accuracy Score: {accuracy_score(Y_test, Y_pred)}")
    else:
        st.warning("Please upload a dataset to continue.")

elif page == "AutoML for Regression":
    st.title("AutoML for Regression App Page")
    if data is not None:
        st.write("Dataset:")
        st.write(data)
        st.write("Dataset Shape:")
        st.write(data.shape)

        st.subheader("AutoML for Regression")

        # Define the maximum allowed dataset size for regression
        max_rows_for_regression = 5000
        max_columns_for_regression = 50

        if data.shape[0] > max_rows_for_regression or data.shape[1] > max_columns_for_regression:
            st.error(f"Dataset size exceeds the maximum allowed for regression (max rows: {max_rows_for_regression}, max columns: {max_columns_for_regression}).")
        else:
            target_variable = st.selectbox("Select Target Variable (Y)", data.columns)
            st.write("Select X Variables:")
            X_variables = st.multiselect("Select Features (X)", [col for col in data.columns if col != target_variable])

            if not X_variables:
                st.warning("Please select one or more features (X) before running the AutoML for regression.")
            else:
                test_size = st.slider("Select Test Size (Fraction)", 0.1, 0.5, 0.2, 0.01)
                random_state = st.slider("Select Random State", 1, 100, 42, 1)

                X = data[X_variables]
                Y = data[target_variable]

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

                st.write("Regression Models:")
                regression_models = [RandomForestRegressor(), LinearRegression(), Ridge(), Lasso()]
                for model in regression_models:
                    model.fit(X_train, Y_train)
                    Y_pred = model.predict(X_test)
                    st.write(f"Model: {type(model).__name__}")
                    st.write(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}")
    else:
        st.warning("Please upload a dataset to continue.")
        
elif page == "AutoML for Clustering":
    st.title("AutoML for Clustering App Page")
    if data is not None:
        st.write("Dataset:")
        st.write(data)
        st.write("Dataset Shape:")
        st.write(data.shape)

        st.subheader("AutoML for Clustering")

        # Define the maximum allowed dataset size for clustering
        max_rows_for_clustering = 5000
        max_columns_for_clustering = 50

        if data.shape[0] > max_rows_for_clustering or data.shape[1] > max_columns_for_clustering:
            st.error(f"Dataset size exceeds the maximum allowed for clustering (max rows: {max_rows_for_clustering}, max columns: {max_columns_for_clustering}).")
        else:
            num_clusters = st.slider("Select the number of clusters:", 2, 10)

            X = data.dropna()

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
    else:
        st.warning("Please upload a dataset to continue.")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.title("Model Evaluation Page")

    # Define the maximum allowed dataset size for model evaluation
    max_rows_for_evaluation = 5000
    max_columns_for_evaluation = 50

    if data.shape[0] > max_rows_for_evaluation or data.shape[1] > max_columns_for_evaluation:
        st.warning(f"Note: The dataset size exceeds the maximum allowed for model evaluation (max rows: {max_rows_for_evaluation}, max columns: {max_columns_for_evaluation}).")
    else:
        # Select Problem Type
        problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])

        if data is not None:
            st.write("Dataset:")
            st.write(data)
            st.write("Dataset Shape:")
            st.write(data.shape)

            # Select Model
            models = {
                "Random Forest Classifier": RandomForestClassifier(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
            }

            model_name = st.selectbox("Select Model", list(models.keys()))

            if st.button("Evaluate Model"):
                model = models[model_name]

                if problem_type == "Classification":
                    X = data.drop(columns=["target_column"])  # Replace "target_column" with the actual target column name
                    y = data["target_column"]  # Replace "target_column" with the actual target column name
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))

                    st.subheader("Accuracy Score")
                    accuracy = accuracy_score(y_test, y_pred)
                    st.text(accuracy)
                else:
                    X = data.drop(columns=["target_column"])  # Replace "target_column" with the actual target column name
                    y = data["target_column"]  # Replace "target_column" with the actual target column name
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    st.subheader("Mean Squared Error")
                    mse = mean_squared_error(y_test, y_pred)
                    st.text(mse)

                    st.subheader("R-squared (R2) Score")
                    r2 = r2_score(y_test, y_pred)
                    st.text(r2)
        else:
            st.warning("Please upload a dataset in the 'Data Cleaning' step to continue.")
            
# Handle errors and optimize performance
try:
    if data is not None:
        st.write("Data Loaded Successfully")
    else:
        st.write("Data Not Loaded. Please upload a dataset.")
except Exception as e:
    st.error(f"An error occurred: {e}")

