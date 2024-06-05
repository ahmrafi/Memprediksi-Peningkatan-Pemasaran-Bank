import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
# @st.cache
def load_data():
    X_data = pd.read_csv("X_train_res.csv")
    y_data = pd.read_csv("y_test_res.csv", header=None, names=["y"])

    # Ensure the number of samples match
    min_samples = min(X_data.shape[0], y_data.shape[0])
    X_data = X_data.iloc[:min_samples, :]
    y_data = y_data.iloc[:min_samples, :]

    return X_data, y_data

# Preprocess data
def preprocess_data(X_data):
    # Label encoding for categorical columns
    label_encoder = LabelEncoder()
    categorical_cols = X_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X_data[col] = label_encoder.fit_transform(X_data[col])
    return X_data

# Train model with hyperparameter tuning
def train_model(X_data, y_data, algorithm, n_estimators, max_depth):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Choose algorithm
    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    else:
        st.error("Invalid algorithm selected.")
        return None, None

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, f"Accuracy: {accuracy:.2f}\n\n{report}"

# Streamlit UI
def main():
    st.set_page_config(page_title="Dashboard Kita", layout="wide")

    st.write("""
    # Hallo Selamat Datang di Dashboard saya
        
    Perkenalkan kami kelompok 5 
              Ahmad Rafi Syaifudin (22031554030),
              Riva Dian Ardiansyah (22031554043),
              Analicia (22031554007)      

    Kita ingin membuat prediksi atas Pemasaran bank untuk mengetahui deposit pengguna
    """)
    st.title("Deposit Prediction App")

    # Load data
    X_data, y_data = load_data()

    # Debugging information
    st.text("X_data shape:")
    st.write(X_data.shape)

    st.text("y_data shape:")
    st.write(y_data.shape)

    # Preprocess data
    X_data = preprocess_data(X_data)

    # Sidebar
    st.sidebar.title("Settings")
    algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Logistic Regression", "Decision Tree"])

    if algorithm == "Random Forest":
        st.sidebar.title("Random Forest Hyperparameters")
        n_estimators = st.sidebar.slider("Number of trees:", 1, 300, 100)
        max_depth = st.sidebar.slider("Maximum depth of trees:", 1, 20, 10)
    else:
        n_estimators, max_depth = None, None

    # Train and evaluate the model
    model, evaluation_report = train_model(X_data, y_data, algorithm, n_estimators, max_depth)

    if model is not None:
        st.success(f"Model train yang digunakan {algorithm}.")
        # st.text("Evaluation Report:")
        # st.text(evaluation_report)

        # User input for prediction
        st.sidebar.title("Make a Prediction")
        user_input = {}
        for col in X_data.columns:
            user_input[col] = st.sidebar.number_input(col, value=X_data[col].median())

        # Make prediction
        user_data = pd.DataFrame([user_input])
        user_data = preprocess_data(user_data)

        prediction = model.predict(user_data)

        st.sidebar.text("Prediction:")
        st.sidebar.text(prediction[0])

if __name__ == "__main__":
    main()
