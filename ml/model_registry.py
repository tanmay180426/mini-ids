"""Model Registry and Artifact Lifecycle Management Module.

Handles saving, validating, versioning, activating, and loading trained ML models
and preprocessors from disk and SQLite metadata storage.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data") / "mini_ids.db"
MODELS_DIR = Path("data") / "models"


@dataclass
class ModelArtifactMeta:
    """Detailed registry record for a saved model artifact bundle."""
    name: str
    version: str
    model_type: str  # 'supervised_binary', 'supervised_multiclass', 'anomaly_isolation_forest'
    algorithm: str
    dataset_name: str
    task_type: str
    feature_schema: List[str]
    metrics: Dict[str, Any]
    artifact_dir: str
    model_path: str
    preprocessor_path: str
    created_at: float
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Manages model artifacts in `data/models/` and records metadata in SQLite."""

    def __init__(self, models_dir: Path = MODELS_DIR, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = Path(db_path)

    def save_model_bundle(
        self,
        name: str,
        version: str,
        model: Any,
        preprocessor: Any,
        model_type: str,
        algorithm: str,
        dataset_name: str,
        task_type: str,
        feature_schema: List[str],
        metrics: Dict[str, Any],
        set_active: bool = False,
    ) -> ModelArtifactMeta:
        """Serialize model, preprocessor, and metadata into a versioned artifact folder."""
        version_dir = self.models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_file = version_dir / "model.joblib"
        preproc_file = version_dir / "preprocessor.joblib"
        meta_file = version_dir / "metadata.json"

        # 1. Serialize model and preprocessor safely
        joblib.dump(model, model_file)
        if preprocessor is not None:
            joblib.dump(preprocessor, preproc_file)

        meta = ModelArtifactMeta(
            name=name,
            version=version,
            model_type=model_type,
            algorithm=algorithm,
            dataset_name=dataset_name,
            task_type=task_type,
            feature_schema=feature_schema,
            metrics=metrics,
            artifact_dir=str(version_dir),
            model_path=str(model_file),
            preprocessor_path=str(preproc_file) if preprocessor is not None else "",
            created_at=time.time(),
            is_active=set_active,
        )

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)

        # 2. Persist to SQLite models table if database exists
        try:
            from storage.database import DatabaseManager
            with DatabaseManager(self.db_path) as db:
                db.init_db()
                db.insert_model({
                    "name": name,
                    "version": version,
                    "model_type": model_type,
                    "algorithm": algorithm,
                    "dataset_name": dataset_name,
                    "task_type": task_type,
                    "metrics": metrics,
                    "feature_schema": feature_schema,
                    "artifact_path": str(version_dir),
                    "created_at": meta.created_at,
                    "is_active": set_active,
                })
                if set_active:
                    db.set_active_model(version, is_active=True)
        except Exception as db_err:
            logger.warning("Could not write model metadata to DB (%s): %s", self.db_path, db_err)

        return meta

    def validate_artifact(self, version_or_dir: Union[str, Path]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Validate artifact integrity before loading into inference pipelines."""
        path = self.models_dir / str(version_or_dir) if not Path(version_or_dir).is_dir() else Path(version_or_dir)
        if not path.is_dir():
            return False, f"Artifact directory not found: {path}", None

        model_file = path / "model.joblib"
        meta_file = path / "metadata.json"

        if not model_file.is_file():
            return False, f"Missing model binary at {model_file}", None
        if not meta_file.is_file():
            return False, f"Missing metadata file at {meta_file}", None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as err:
            return False, f"Corrupted metadata JSON: {err}", None

        # Verify safe loading with a dry-run test
        try:
            loaded_model = joblib.load(model_file)
            feature_schema = meta.get("feature_schema", [])
            if feature_schema:
                # Test dummy vector
                dummy = np.zeros((1, len(feature_schema)))
                if hasattr(loaded_model, "predict"):
                    _ = loaded_model.predict(dummy)
        except Exception as dry_err:
            return False, f"Failed dry-run validation inference: {dry_err}", None

        return True, "Artifact verified and schema validated successfully.", meta

    def load_active_bundle(
        self,
        model_type: Optional[str] = None,
    ) -> Optional[Tuple[Any, Any, Dict[str, Any]]]:
        """Load the active model and preprocessor bundle from disk and database."""
        target_version = None

        # Query database for active version
        try:
            from storage.database import DatabaseManager
            with DatabaseManager(self.db_path) as db:
                db.init_db()
                active_record = db.get_active_model(model_type)
                if active_record:
                    target_version = active_record.get("version")
        except Exception as err:
            logger.debug("Database query for active model failed: %s", err)

        # Fallback: scan disk for most recent verified model
        if not target_version:
            versions = sorted(
                [d.name for d in self.models_dir.iterdir() if d.is_dir() and (d / "model.joblib").exists()],
                reverse=True,
            )
            if not versions:
                return None
            target_version = versions[0]

        valid, msg, meta = self.validate_artifact(target_version)
        if not valid or meta is None:
            logger.warning("Active artifact %s failed validation: %s", target_version, msg)
            return None

        version_dir = self.models_dir / target_version
        model = joblib.load(version_dir / "model.joblib")
        preprocessor = None
        if (version_dir / "preprocessor.joblib").exists():
            preprocessor = joblib.load(version_dir / "preprocessor.joblib")

        return model, preprocessor, meta
