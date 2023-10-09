import streamlit as st
import pandas as pd
import numpy as np
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import KMeans
from apyori import apriori  # Import apyori for association rule mining

# Title and Description
st.title("AutoML App with Streamlit")

# Multi-page navigation
app_mode = st.sidebar.selectbox("Select a page:", ["Data Visualization", "AutoML Tasks"])

@st.cache
def load_example_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        return datasets.load_iris(as_frame=True).frame
    elif dataset_name == "Diabetes Dataset":
        return datasets.load_diabetes(as_frame=True).frame

@st.cache
def load_uploaded_dataset(uploaded_file):
    return pd.read_csv(uploaded_file)

def mine_association_rules(data, min_support, min_confidence):
    # Perform association rule mining
    association_results = list(apriori(data, min_support=min_support, min_confidence=min_confidence))

    return association_results

if app_mode == "Data Visualization":
    st.write("## Data Visualization")

    # Dataset selection
    dataset_option = st.radio("Select a dataset source:", ["Use Example Dataset", "Upload CSV Dataset"])
    uploaded_file = None

    if dataset_option == "Upload CSV Dataset":
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if dataset_option == "Use Example Dataset" or (uploaded_file is not None):
        try:
            if dataset_option == "Use Example Dataset":
                # Use example datasets
                example_datasets = {
                    "Iris Dataset": "Iris Dataset",
                    "Diabetes Dataset": "Diabetes Dataset",
                }
                selected_dataset_name = st.selectbox("Select an example dataset:", list(example_datasets.keys()))
                data = load_example_dataset(example_datasets[selected_dataset_name])
            else:
                # Use uploaded dataset
                data = load_uploaded_dataset(uploaded_file)

            # Display dataset details
            st.write("Dataset description:")
            st.write(data.describe())

            # Automatic data visualization
            st.subheader("Automatic Data Visualization")

            # Pairwise correlation heatmap
            st.write("### Pairwise Correlation Heatmap")
            st.write("Correlation between features:")
            corr_matrix = data.corr()
            st.write(corr_matrix.style.background_gradient(cmap="coolwarm", axis=None))

            # Visualize numerical features
            numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            for column in numerical_columns:
                st.subheader(f"### Visualizations for {column}")

                # Scatter plot for numerical features
                st.write(f"#### Scatter Plot for {column}")
                st.scatter_chart(data[[column]])

                # Box plot for numerical features
                st.write(f"#### Box Plot for {column}")
                st.box_chart(data[[column]])

                # Line plot for time series data (assuming an index as the time axis)
                if column == data.index.name:
                    st.write(f"#### Line Plot for {column}")
                    st.line_chart(data[[column]])

            # Count plot for categorical features
            st.write("### Count Plot for Categorical Features")
            categorical_columns = data.select_dtypes(include=[np.object]).columns.tolist()
            for column in categorical_columns:
                st.subheader(f"{column} Count Plot")
                st.bar_chart(data[column].value_counts())

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    else:
        st.write("Upload a CSV file or select an example dataset to visualize the data.")

elif app_mode == "AutoML Tasks":
    st.write("## AutoML Tasks")

    # Task selection
    task = st.selectbox("Select a task:", ["Classification", "Regression", "Clustering", "Association"])

    # Dataset selection
    dataset_option = st.radio("Select a dataset source:", ["Use Example Dataset", "Upload CSV Dataset"])
    uploaded_file = None

    if dataset_option == "Upload CSV Dataset":
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if dataset_option == "Use Example Dataset" or (uploaded_file is not None):
        try:
            if dataset_option == "Use Example Dataset":
                # Use example datasets
                if task == "Classification":
                    data = load_example_dataset("Iris Dataset")
                elif task == "Regression":
                    data = load_example_dataset("Diabetes Dataset")
                elif task == "Clustering":
                    data = load_example_dataset("Iris Dataset")
                elif task == "Association":
                    data = [
                        ["milk", "bread", "nuts", "apples"],
                        ["milk", "bread", "diapers", "eggs"],
                        ["milk", "eggs", "yogurt"],
                        ["bread", "nuts", "apples"],
                        ["milk", "bread", "nuts", "diapers", "eggs"],
                        ["eggs", "diapers", "yogurt"],
                        ["milk", "apples", "yogurt"],
                        ["bread", "nuts", "yogurt"],
                        ["bread", "nuts", "apples", "yogurt"],
                        ["milk", "bread", "nuts", "apples", "yogurt"],
                        ["milk", "diapers"],
                        ["bread", "nuts", "diapers"],
                        ["milk", "bread", "nuts", "apples", "diapers"],
                        ["milk", "eggs", "yogurt"],
                        ["milk", "bread", "nuts", "eggs", "yogurt"],
                    ]

            else:
                # Use uploaded dataset
                data = load_uploaded_dataset(uploaded_file)

            # Check if there are at least two columns
            if isinstance(data, pd.DataFrame) and data.shape[1] < 2:
                st.error("The dataset must have at least two columns.")

            # Display dataset details
            if isinstance(data, pd.DataFrame):
                st.write("Dataset description:")
                st.write(data.describe())

            if task == "Classification":
                # Classification: User selects target column and trains a classification model
                st.subheader("Classification Task")
                target_column = st.selectbox("Select the target column:", data.columns)
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Check if the target column exists
                if target_column not in data.columns:
                    st.error("Selected target column does not exist in the dataset.")
                else:
                    # Perform one-hot encoding for categorical features
                    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
                    if categorical_features:
                        X_encoded = pd.get_dummies(X, columns=categorical_features)
                    else:
                        X_encoded = X

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42
                    )

                    # AutoML model training with reduced time limit
                    tpot_classifier = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
                    tpot_classifier.fit(X_train, y_train)
                    st.write("AutoML classification model training completed!")

                    # Model evaluation
                    y_pred_classifier = tpot_classifier.predict(X_test)
                    accuracy_classifier = accuracy_score(y_test, y_pred_classifier)
                    st.write(f"Accuracy: {accuracy_classifier:.2f}")

            elif task == "Regression":
                # Regression: User selects target column and trains a regression model
                st.subheader("Regression Task")
                target_column = st.selectbox("Select the target column:", data.columns)
                X = data.drop(columns=[target_column])
                y = data[target_column]

                # Check if the target column exists
                if target_column not in data.columns:
                    st.error("Selected target column does not exist in the dataset.")
                else:
                    # Perform one-hot encoding for categorical features
                    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
                    if categorical_features:
                        X_encoded = pd.get_dummies(X, columns=categorical_features)
                    else:
                        X_encoded = X

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_encoded, y, test_size=0.2, random_state=42
                    )

                    # AutoML model training with reduced time limit
                    tpot_regressor = TPOTRegressor(generations=5, population_size=20, verbosity=2, random_state=42)
                    tpot_regressor.fit(X_train, y_train)
                    st.write("AutoML regression model training completed!")

                    # Model evaluation
                    y_pred_regressor = tpot_regressor.predict(X_test)
                    mse_regressor = mean_squared_error(y_test, y_pred_regressor)
                    st.write(f"Mean Squared Error: {mse_regressor:.2f}")

            elif task == "Clustering":
                # Clustering: User selects the number of clusters and performs K-Means clustering
                st.subheader("Clustering Task")
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

            elif task == "Association":
                # Association: User selects minimum support and confidence values to discover association rules
                st.subheader("Association Task")
                min_support = st.slider("Select minimum support:", 0.01, 0.5, 0.1)
                min_confidence = st.slider("Select minimum confidence:", 0.1, 1.0, 0.5)

                # Check if the dataset has enough rows for association rule mining
                if isinstance(data, list) and len(data) < 2:
                    st.error("The dataset must have at least two rows for association rule mining.")
                else:
                    # Perform association rule mining
                    association_results = mine_association_rules(data, min_support, min_confidence)

                    st.write("Discovered Association Rules:")
                    for item in association_results:
                        pair = item[0]
                        items = [x for x in pair]
                        st.write(f"Rule: {items[0]} -> {items[1]}")
                        st.write(f"Support: {item[1]:.2f}")
                        st.write(f"Confidence: {item[2][0][2]:.2f}")
                        st.write(f"Lift: {item[2][0][3]:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    else:
        st.write("Upload a CSV file or select an example dataset to perform AutoML tasks.")

# End the Streamlit app
