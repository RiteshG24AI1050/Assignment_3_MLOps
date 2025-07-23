import numpy as np
import pandas as pd
import joblib
import os
import logging
from datetime import datetime
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


MODEL_DIR = "models"
MODEL_NAME = "linear_model.joblib"
TEST_DATA_NAME = "test_data.joblib"
SEED = 42
TEST_SIZE = 0.2

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def load_and_prepare_data():
    try:
        logging.info("Loading California Housing dataset...")
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        logging.info(f"Dataset shape: {X.shape}, Target shape: {y.shape}")
        logging.info(f"Features: {housing.feature_names}")
        return X, y
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        raise


def train_model(X, y):
    try:
        logging.info("Splitting the dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED
        )

        logging.info("Training the Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        logging.info(f"Training R² Score: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        logging.info(f"Testing  R² Score: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        return model, (X_test, y_test)
    except Exception as e:
        logging.error(f"Model training failed: {e}")
        raise


def save_model(model, test_data):
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_path = os.path.join(MODEL_DIR, f"{timestamp}_{MODEL_NAME}")
        test_path = os.path.join(MODEL_DIR, f"{timestamp}_{TEST_DATA_NAME}")

        joblib.dump(model, model_path)
        joblib.dump(test_data, test_path)

        logging.info(f"Model saved at {model_path}")
        logging.info(f"Test data saved at {test_path}")

        logging.debug(f"Model coefficients shape: {model.coef_.shape}")
        logging.debug(f"Model intercept: {model.intercept_}")
        logging.debug(f"Model coefficients: {model.coef_}")
    except Exception as e:
        logging.error(f"Failed to save model or test data: {e}")
        raise


def main():
    logging.info("Starting California Housing model training...")

    X, y = load_and_prepare_data()
    model, test_data = train_model(X, y)
    save_model(model, test_data)

    logging.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()