import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

# ---------------------------------------------
# NEW imports from evidently.project submodules
# ---------------------------------------------
from evidently.project import Project
from evidently.project import ColumnMapping
from evidently.project.dashboard import Dashboard
from evidently.project.tabs import DataDriftTab, ClassificationPerformanceTab

REF_DATA_PATH = "/data/ref_data.csv"
PROD_DATA_PATH = "/data/prod_data.csv"

def build_static_report():
    """
    Optionally generate a static HTML report using the older 'Report' class.
    """
    ref_data = pd.read_csv(REF_DATA_PATH)
    prod_data = pd.read_csv(PROD_DATA_PATH)

    # If your columns follow the "PCA_1, PCA_2, ..., target, prediction" pattern:
    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=[c for c in ref_data.columns if c.startswith("PCA_")],
    )

    report = Report(
        metrics=[
            DataDriftPreset(),
            ClassificationPreset(
                target="target",
                prediction="prediction"
            ),
        ]
    )
    report.run(reference_data=ref_data, current_data=prod_data, column_mapping=column_mapping)
    report.save_html("report.html")
    print("Static report has been generated: report.html")

def build_evidently_project():
    """
    Use the new Project approach to create a 'live' Evidently UI with evidently ui command.
    """
    ref_data = pd.read_csv(REF_DATA_PATH)
    prod_data = pd.read_csv(PROD_DATA_PATH)

    column_mapping = ColumnMapping(
        target="target",
        prediction="prediction",
        numerical_features=[c for c in ref_data.columns if c.startswith("PCA_")],
    )

    project = Project(
        dashboards=[
            Dashboard(
                tabs=[
                    DataDriftTab(),
                    ClassificationPerformanceTab()
                ]
            )
        ]
    )

    # Add a dataset pair to the project
    project.calculate(
        reference_data=ref_data,
        current_data=prod_data,
        column_mapping=column_mapping,
        dataset_name="my_sign_language_data",
    )

    # Save the project’s results to 'evidently_project/' for evidently ui
    project.save("evidently_project")
    print("Evidently project data saved to 'evidently_project/'")

def main():
    # 1) Optional static HTML
    build_static_report()
    # 2) Build the Project for the new evidently UI
    build_evidently_project()

if __name__ == "__main__":
    main()
