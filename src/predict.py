import numpy as np
import joblib
import os
import logging
from sklearn.metrics import r2_score, mean_squared_error
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

MODEL_DIR = "models"
MODEL_NAME = "linear_model.joblib"
TEST_DATA_NAME = "test_data.joblib"
MIN_R2_THRESHOLD = 0.5


def load_model_and_data():
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    test_data_path = os.path.join(MODEL_DIR, TEST_DATA_NAME)

    logging.info("Loading model from: %s", model_path)
    logging.info("Loading test data from: %s", test_data_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data file not found at: {test_data_path}")

    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    logging.info(f"Model and test data loaded successfully.")
    logging.info(f"Test data shape — X: {X_test.shape}, y: {y_test.shape}")
    return model, X_test, y_test


def run_predictions(model, X_test, y_test):
    logging.info("Running model predictions...")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    logging.info(f"R² Score: {r2:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")

    logging.info("\nSample Predictions:")
    logging.info("Actual\tPredicted")
    for i in range(min(5, len(y_test))):
        logging.info(f"{y_test[i]:.2f}\t{y_pred[i]:.2f}")

    return r2, rmse


def main():
    logging.info("Starting container prediction verification...")

    try:
        model, X_test, y_test = load_model_and_data()
        r2, _ = run_predictions(model, X_test, y_test)

        if r2 >= MIN_R2_THRESHOLD:
            logging.info(f"\n Container verification PASSED (R² = {r2:.4f})")
            sys.exit(0)
        else:
            logging.error(f"\n Container verification FAILED (R² = {r2:.4f} < {MIN_R2_THRESHOLD})")
            sys.exit(1)

    except Exception as e:
        logging.exception(f"\n Container verification FAILED with error: {str(e)}")
        sys.exit(1)

    logging.info("Verification complete.")


if __name__ == "__main__":
    main()