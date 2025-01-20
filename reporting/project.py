import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.ui.workspace import Workspace  # Workspace used here
import pickle
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score



# Define workspace path (local directory)
WORKSPACE_PATH = "./reporting/evidently_ui_workspace"

# Create a local workspace if it doesn't exist
workspace = Workspace.create(WORKSPACE_PATH)

# Define the paths for your reference and production data
REF_DATA_PATH = "../data/ref_data.csv"
PROD_DATA_PATH = "../data/prod_data.csv"

# Define the model path
MODEL_PATH = "../artifacts/model_xgb.pkl"

def debug_label_types(df: pd.DataFrame, name: str):
    """
    Helper function to scan 'target' and 'prediction' columns for any non-string values.
    Prints them if found (for debugging issues like "TypeError: '<' not supported").
    """
    unique_vals = set(df['target'].unique())
    if 'prediction' in df.columns:
        unique_vals.update(df['prediction'].unique())

    for val in unique_vals:
        if not isinstance(val, str):
            print(f"[DEBUG] {name}: Found non-string value => {repr(val)} (type={type(val)})")

def build_static_report():
    """
    Generate a static HTML file (report.html) comparing reference vs. production data.
    """
    # 1. Load data
    ref_data = pd.read_csv(REF_DATA_PATH)
    prod_data = pd.read_csv(PROD_DATA_PATH)

    # 2. Drop any "Unnamed" columns (common when CSVs have trailing commas)
    ref_data = ref_data.loc[:, ~ref_data.columns.str.contains("^Unnamed")]
    prod_data = prod_data.loc[:, ~prod_data.columns.str.contains("^Unnamed")]

    print("Reference data shape BEFORE rename:", ref_data.shape)
    print("Production data shape:", prod_data.shape)

    # 3. Rename reference columns from 0..99,label → PCA_1..PCA_100,target
    ref_columns = [f"PCA_{i+1}" for i in range(100)] + ["target"]
    ref_data.columns = ref_columns
    print("Reference data shape AFTER rename:", ref_data.shape)
    print("Reference columns:", ref_data.columns.tolist())
    print("Production columns:", prod_data.columns.tolist())

    # 4. Ensure reference has a "prediction" column, even if dummy
    if "prediction" not in ref_data.columns:
        ref_data["prediction"] = "dummy_pred"

    # If production does not have "prediction" column, add one too
    if "prediction" not in prod_data.columns:
        prod_data["prediction"] = "dummy_pred"

    # 5. Convert PCA columns to numeric (coerce => NaN), then drop rows that fail
    pca_cols = [f"PCA_{i+1}" for i in range(100)]
    for col in pca_cols:
        ref_data[col] = pd.to_numeric(ref_data[col], errors="coerce")
        prod_data[col] = pd.to_numeric(prod_data[col], errors="coerce")

    ref_data.dropna(subset=pca_cols, inplace=True)
    prod_data.dropna(subset=pca_cols, inplace=True)

    # 6. Force target & prediction to string
    ref_data["target"] = ref_data["target"].apply(str)
    ref_data["prediction"] = ref_data["prediction"].apply(str)
    prod_data["target"] = prod_data["target"].apply(str)
    prod_data["prediction"] = prod_data["prediction"].apply(str)

    # 7. Optional debug checks
    print("\n--- Debug: Checking for non-string values in reference ---")
    debug_label_types(ref_data, name="Reference")
    print("\n--- Debug: Checking for non-string values in production ---")
    debug_label_types(prod_data, name="Production")
    print()

    # 8. Column mapping
    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=pca_cols,
        categorical_features=None,
        datetime_features=None
    )

    # 9. Create the Evidently report
    report = Report(
        metrics=[
            DataDriftPreset(),
            # Add more metrics like ClassificationPreset if needed
        ]
    )

    # 10. Run the report
    report.run(
        reference_data=ref_data,
        current_data=prod_data,
        column_mapping=column_mapping,
    )

    # 11. Create a project in the workspace if necessary
    project = workspace.create_project("Data Drift Project")  # Create a new project with a name
    project.description = "This is a test project to track data drift."

    # 12. Add the report to the workspace under the created project
    workspace.add_report(project.id, report)  # Add the report to the project using its ID

    # 13. Save HTML report to file
    report.save_html("./report.html")
    print("[INFO] Static report has been generated: report.html")


def preprocess_data():
    """Load and preprocess reference and production datasets."""
    ref_data = pd.read_csv(REF_DATA_PATH)
    prod_data = pd.read_csv(PROD_DATA_PATH)

    # Drop unnecessary columns
    ref_data = ref_data.loc[:, ~ref_data.columns.str.contains("^Unnamed")]
    prod_data = prod_data.loc[:, ~prod_data.columns.str.contains("^Unnamed")]

    # Rename columns
    ref_columns = [f"PCA_{i+1}" for i in range(100)] + ["target"]
    ref_data.columns = ref_columns

    # Convert PCA columns to numeric
    pca_cols = [f"PCA_{i+1}" for i in range(100)]
    for col in pca_cols:
        ref_data[col] = pd.to_numeric(ref_data[col], errors="coerce")
        prod_data[col] = pd.to_numeric(prod_data[col], errors="coerce")

    ref_data.dropna(subset=pca_cols, inplace=True)
    prod_data.dropna(subset=pca_cols, inplace=True)

    # Convert target to string
    ref_data["target"] = ref_data["target"].apply(str)
    prod_data["target"] = prod_data["target"].apply(str)

    return ref_data, prod_data, pca_cols

def load_model():
    """Load the trained model from the specified path."""
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    return model

def evaluate_model(ref_data, prod_data, pca_cols, model):
    """Generate predictions and evaluate model performance."""
    ref_data["prediction"] = model.predict(ref_data[pca_cols])
    prod_data["prediction"] = model.predict(prod_data[pca_cols])

    # Compute evaluation metrics
    metrics = {
        "F1 Score": f1_score(ref_data["target"], ref_data["prediction"], average="weighted"),
        "Balanced Accuracy": balanced_accuracy_score(ref_data["target"], ref_data["prediction"]),
        "Recall (Rappel)": recall_score(ref_data["target"], ref_data["prediction"], average="weighted"),
        "Precision": precision_score(ref_data["target"], ref_data["prediction"], average="weighted"),
    }

    print("\n[INFO] Model Performance on Reference Data:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return ref_data, prod_data

def build_performance_dashboard(ref_data, prod_data, pca_cols):
    """Create and save the Evidently performance dashboard."""
    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=pca_cols,
    )

    # Create Evidently report
    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=ref_data, current_data=prod_data, column_mapping=column_mapping)

    # Add to workspace
    project = workspace.create_project("Model Performance Project")
    workspace.add_report(project.id, report)

    # Save as HTML
    report.save_html("./model_performance_dashboard.html")
    print("[INFO] Model performance dashboard saved: model_performance_dashboard.html")


def main():
    # Call the function to generate the report
    # Load and preprocess data
    ref_data, prod_data, pca_cols = preprocess_data()

    # Load the model
    model = load_model()

    # Evaluate model performance
    ref_data, prod_data = evaluate_model(ref_data, prod_data, pca_cols, model)

    # Create performance dashboard
    build_performance_dashboard(ref_data, prod_data, pca_cols)
    
    build_static_report()

if __name__ == "__main__":
    main()
