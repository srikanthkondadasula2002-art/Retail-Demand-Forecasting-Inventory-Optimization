"""
BigQuery Configuration & Connection Manager
Handles project credentials, dataset identifiers, and client initialization.
"""

import os
from typing import Optional
from google.cloud import bigquery
from google.oauth2 import service_account

# Configuration defaults
DEFAULT_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "retail-demand-forecasting-dwh")
DEFAULT_DATASET_ID = os.getenv("BIGQUERY_DATASET_ID", "retail_demand_dwh")
DEFAULT_LOCATION = os.getenv("BIGQUERY_LOCATION", "US")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", None)

class BigQueryConfig:
    def __init__(
        self,
        project_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials_path: Optional[str] = None
    ):
        self.project_id = project_id or DEFAULT_PROJECT_ID
        self.dataset_id = dataset_id or DEFAULT_DATASET_ID
        self.location = location or DEFAULT_LOCATION
        self.credentials_path = credentials_path or CREDENTIALS_PATH

    @property
    def full_dataset_id(self) -> str:
        return f"{self.project_id}.{self.dataset_id}"

    def get_table_id(self, table_name: str) -> str:
        return f"{self.full_dataset_id}.{table_name}"

    def get_client(self, require_auth: bool = False) -> Optional[bigquery.Client]:
        """
        Initializes Google Cloud BigQuery client.
        Supports explicit service account JSON, ADC, or graceful None if unconfigured.
        """
        try:
            if self.credentials_path and os.path.exists(self.credentials_path):
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                return bigquery.Client(
                    project=self.project_id,
                    credentials=credentials,
                    location=self.location
                )
            # Attempt default credentials
            return bigquery.Client(project=self.project_id, location=self.location)
        except Exception as e:
            if require_auth:
                raise RuntimeError(
                    f"Failed to authenticate with BigQuery. Please set GOOGLE_APPLICATION_CREDENTIALS "
                    f"or run 'gcloud auth application-default login'. Error: {e}"
                )
            return None
