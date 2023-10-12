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
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report,  f1_score, mean_absolute_error, r2_score
from sklearn.svm import SVC, SVR

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
page = st.sidebar.radio("**Select a Page**", ["Home Page", "Data Cleaning", "Data Encoding", "Data Visualization", "ML Model Selection", "AutoML for Classification", "AutoML for Regression", "AutoML for Clustering", "Model Evaluation"])

# Introduction Page
if page == "Home Page":
    st.title("Introduction")

    st.header("Welcome to the AutoML Application!")
    st.write("This application is designed to help you streamline the process of data analysis and machine learning model selection. Follow the steps below to make the most of this application:")

    st.subheader("Home Page")
    st.markdown(
        "The **Home Page** is the starting point of the application. You can navigate to different sections of the app using the sidebar navigation."
    )

    st.subheader("Data Cleaning Page")
    st.markdown(
        "The **Data Cleaning Page** allows you to clean missing values in your dataset. Select a dataset and apply different cleaning techniques to handle missing values."
    )

    st.subheader("Data Encoding Page")
    st.markdown(
        "The **Data Encoding Page** empowers you to encode categorical variables in the dataset. Select a dataset and apply different encoding techniques to handle categorical variables."
    )

    st.subheader("Data Visualization Page")
    st.markdown(
        "The **Data Visualization Page** lets you visualize the dataset using various techniques such as histograms, scatter plots, and heat maps. Select a dataset and explore its visual representations."
    )

    st.subheader("ML Model Selection Page")
    st.markdown(
        "The **ML Model Selection Page** helps you choose the right machine learning model based on the problem type (classification, regression, or time series). Pick a dataset and select the target variable to find the best machine learning model."
    )

    st.subheader("AutoML for Regression Page")
    st.markdown(
        "The **AutoML for Regression Page** enables you to perform automated machine learning (AutoML) for regression problems using the `lazyRegression` library. Select a dataset, choose the target variable, and run the AutoML algorithm."
    )

    st.subheader("AutoML for Classification Page")
    st.markdown(
        "The **AutoML for Classification Page** allows you to perform automated machine learning (AutoML) for classification problems using the `lazyClassifier` library. Select a dataset, choose the target variable, and run the AutoML algorithm."
    )

    st.subheader("Model Evaluation Page")
    st.markdown(
        "The **Model Evaluation Page** is where you can evaluate machine learning models on your dataset. Choose the problem type (classification or regression), select X and Y variables, and pick a model to see evaluation results."
    )

    st.subheader("Using the App")
    st.markdown(
        "1. Start on the **Home Page**, and then navigate to the pages that match your needs."
    )
    st.markdown(
        "2. Ensure you upload a dataset in the **Data Cleaning Page** before proceeding to other pages that require a dataset."
    )
    st.markdown(
        "3. Use the sidebar navigation to switch between pages and follow the instructions on each page to complete the tasks."
    )
    st.markdown(
        "4. Make sure to select the appropriate problem type (classification or regression) and follow any additional instructions for model selection and evaluation."
    )

    st.subheader("Additional Tips")
    st.markdown(
        "5. You can always refer back to the **Introduction** page for a quick overview of the app's functionality and how to use it."
    )
    st.markdown(
        "6. Don't hesitate to reach out for assistance or if you have any questions about using the application effectively."
    )

    st.markdown(
        "Enjoy using the AutoML Application and have a productive data analysis and model selection experience!"
    )

# Data Cleaning Page
elif page == "Data Cleaning":
    st.title("Data Cleaning Page")

    # Check if the dataset is available
    if data is not None:
        st.write("Dataset:")
        st.write(data)
        st.write("Dataset Shape:")
        st.write(data.shape)

        # Check if the dataset has too many rows or columns
        max_rows_for_cleaning = 3500
        max_columns_for_cleaning = 50

        if data.shape[0] > max_rows_for_cleaning or data.shape[1] > max_columns_for_cleaning:
            st.warning(f"Note: The dataset size exceeds the maximum allowed for data cleaning (max rows: {max_rows_for_cleaning}, max columns: {max_columns_for_cleaning}).")
        else:
            # Check if there are categorical features
            categorical_features = data.select_dtypes(include=['object']).columns.tolist()

            if not categorical_features:
                st.warning("Note: The dataset has no categorical features, so you cannot use the 'mean' or 'median' methods.")
            else:
                st.write("Categorical Features:")
                st.write(categorical_features)

            # Choose missing value handling method
            st.subheader("Missing Value Handling")
            methods = ["Drop Missing Values", "Custom Value"]
            if categorical_features:
                methods.extend(["Mean", "Median"])

            method = st.selectbox("Select a method:", methods)

            if method == "Drop Missing Values":
                data_cleaned = data.dropna()
                st.write("Dropped missing values.")
                st.write(data_cleaned)

            elif method == "Custom Value":
                custom_value = st.text_input("Enter a custom value to fill missing cells:", "0")
                data_cleaned = data.fillna(custom_value)
                st.write(f"Filled missing values with custom value: {custom_value}")
                st.write(data_cleaned)

            elif method == "Mean" and categorical_features:
                st.warning("Mean method not available due to the presence of categorical features. Use 'Drop Missing Values' or 'Custom Value' instead.")

            elif method == "Median" and categorical_features:
                st.warning("Median method not available due to the presence of categorical features. Use 'Drop Missing Values' or 'Custom Value' instead.")
    else:
        st.warning("Please upload a dataset in the 'Data Cleaning' step to continue.")

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
    max_rows_for_evaluation = 10000
    max_columns_for_evaluation = 100

    if data is not None:
        if data.shape[0] > max_rows_for_evaluation or data.shape[1] > max_columns_for_evaluation:
            st.warning(f"Note: The dataset size exceeds the maximum allowed for model evaluation (max rows: {max_rows_for_evaluation}, max columns: {max_columns_for_evaluation}).")
        else:
            # Select Problem Type
            problem_type = st.radio("Select Problem Type", ["Classification", "Regression"])

            if problem_type == "Classification":
                st.subheader("Classification Model Evaluation")

                classification_models = ["Random Forest Classifier", "Logistic Regression", "Support Vector Machine"]
                selected_model = st.selectbox("Select a Classification Model", classification_models)

                model = None
                if selected_model == "Random Forest Classifier":
                    model = RandomForestClassifier()
                elif selected_model == "Logistic Regression":
                    model = LogisticRegression()
                elif selected_model == "Support Vector Machine":
                    model = SVC()

                if model is not None:
                    # Get X and Y variable names from the user using select boxes
                    x_variable = st.selectbox("Select the X variable", data.columns)
                    y_variable = st.selectbox("Select the Y variable", data.columns)

                    # Validate variable names and perform data splitting
                    if x_variable != y_variable:
                        X = data[[x_variable]]
                        y = data[y_variable]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # Calculate evaluation metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted")

                        st.write(f"Selected Classification Model: {selected_model}")
                        st.write(f"X Variable: {x_variable}")
                        st.write(f"Y Variable: {y_variable}")
                        st.write(f"Accuracy: {accuracy:.2f}")
                        st.write(f"F1 Score: {f1:.2f}")

                    else:
                        st.error("X and Y variable names cannot be the same.")
                else:
                    st.error("An error occurred while selecting the model. Please try again.")

            elif problem_type == "Regression":
                st.subheader("Regression Model Evaluation")

                regression_models = ["Random Forest Regressor", "Linear Regression", "Support Vector Machine"]
                selected_model = st.selectbox("Select a Regression Model", regression_models)

                model = None
                if selected_model == "Random Forest Regressor":
                    model = RandomForestRegressor()
                elif selected_model == "Linear Regression":
                    model = LinearRegression()
                elif selected_model == "Support Vector Machine":
                    model = SVR()

                if model is not None:
                    # Get X and Y variable names from the user using select boxes
                    x_variable = st.selectbox("Select the X variable", data.columns)
                    y_variable = st.selectbox("Select the Y variable", data.columns)

                    # Validate variable names and perform data splitting
                    if x_variable != y_variable:
                        X = data[[x_variable]]
                        y = data[y_variable]

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # Calculate evaluation metrics
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        st.write(f"Selected Regression Model: {selected_model}")
                        st.write(f"X Variable: {x_variable}")
                        st.write(f"Y Variable: {y_variable}")
                        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                        st.write(f"R-squared (R^2): {r2:.2f}")

                    else:
                        st.error("X and Y variable names cannot be the same.")
                else:
                    st.error("An error occurred while selecting the model. Please try again.")
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

