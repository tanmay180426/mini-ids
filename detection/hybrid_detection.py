"""Hybrid Intrusion Detection Engine.

Combines stateful heuristic rule detectors (PortScan, SynRate, TrafficAnomaly)
with flow-level supervised machine learning classifiers and unsupervised Isolation
Forest anomaly detectors into a unified, resilient detection pipeline.
"""

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from capture.packet_capture import NormalizedPacket
from detection import (
    Alert,
    PortScanDetector,
    SynRateDetector,
    TrafficAnomalyDetector,
)

if TYPE_CHECKING:
    from ml.feature_extraction import FlowFeatureExtractor, NetworkFlow

logger = logging.getLogger(__name__)


class HybridDetectionEngine:
    """Coordinates heuristic and machine-learning detection engines."""

    def __init__(
        self,
        port_scan_detector: Optional[PortScanDetector] = None,
        syn_rate_detector: Optional[SynRateDetector] = None,
        traffic_anomaly_detector: Optional[TrafficAnomalyDetector] = None,
        supervised_model: Optional[Any] = None,
        supervised_preprocessor: Optional[Any] = None,
        supervised_meta: Optional[Dict[str, Any]] = None,
        anomaly_model: Optional[Any] = None,
        anomaly_preprocessor: Optional[Any] = None,
        anomaly_meta: Optional[Dict[str, Any]] = None,
        flow_timeout: float = 15.0,
        enable_heuristic: bool = True,
        enable_supervised: bool = True,
        enable_anomaly: bool = True,
    ) -> None:
        # 1. Heuristic Detectors
        self.enable_heuristic = enable_heuristic
        self.port_scan = port_scan_detector or PortScanDetector()
        self.syn_rate = syn_rate_detector or SynRateDetector()
        self.traffic_anomaly = traffic_anomaly_detector or TrafficAnomalyDetector()

        # 2. Flow Extractor for ML
        from ml.feature_extraction import FlowFeatureExtractor
        self.flow_extractor = FlowFeatureExtractor(flow_timeout=flow_timeout)

        # 3. Supervised Model
        self.enable_supervised = enable_supervised
        self.supervised_model = supervised_model
        self.supervised_preprocessor = supervised_preprocessor
        self.supervised_meta = supervised_meta or {}

        # 4. Anomaly Model
        self.enable_anomaly = enable_anomaly
        self.anomaly_model = anomaly_model
        self.anomaly_preprocessor = anomaly_preprocessor
        self.anomaly_meta = anomaly_meta or {}

    def process_packet(self, packet: NormalizedPacket) -> Optional[Alert]:
        """Process a single packet and return the first alert if triggered (standard detector interface)."""
        alerts, _ = self.process_packet_all(packet)
        return alerts[0] if alerts else None

    def process_packet_all(self, packet: NormalizedPacket) -> Tuple[List[Alert], List["NetworkFlow"]]:
        """Process a single packet through heuristic detectors and flow-based ML.

        Returns:
            Tuple[List[Alert], List[NetworkFlow]]:
                - Generated security alerts
                - Completed network flows (if any completed during this step)
        """
        alerts: List[Alert] = []

        # 1. Heuristic Detection
        if self.enable_heuristic:
            for detector in [self.port_scan, self.syn_rate, self.traffic_anomaly]:
                alert = detector.process_packet(packet)
                if alert is not None:
                    # Tag with HEURISTIC detection method
                    alert.detection_method = "HEURISTIC"
                    alerts.append(alert)

        # 2. Flow Aggregation & ML Detection
        completed_flows = self.flow_extractor.process_packet(packet)
        for flow in completed_flows:
            ml_alerts = self._evaluate_flow_ml(flow)
            alerts.extend(ml_alerts)

        return alerts, completed_flows

    def flush(self) -> Tuple[List[Alert], List["NetworkFlow"]]:
        """Flush remaining active flows and evaluate ML on any final flows."""
        remaining_flows = self.flow_extractor.flush()
        alerts: List[Alert] = []
        for flow in remaining_flows:
            ml_alerts = self._evaluate_flow_ml(flow)
            alerts.extend(ml_alerts)
        return alerts, remaining_flows

    def _evaluate_flow_ml(self, flow: "NetworkFlow") -> List[Alert]:
        """Apply supervised and anomaly detection to a completed network flow."""
        alerts: List[Alert] = []
        features = flow.extract_features()

        # A. Supervised ML Detection
        if self.enable_supervised and self.supervised_model is not None and self.supervised_preprocessor is not None:
            try:
                X_vec = self.supervised_preprocessor.transform_inference(features)
                pred = self.supervised_model.predict(X_vec)[0]

                # Check confidence/probability if available
                confidence: Optional[float] = None
                if hasattr(self.supervised_model, "predict_proba"):
                    try:
                        probs = self.supervised_model.predict_proba(X_vec)[0]
                        confidence = float(np.max(probs))
                    except Exception as prob_err:
                        logger.debug("predict_proba failed: %s", prob_err)
                        confidence = None

                # Check if predicted class is an attack
                task_type = self.supervised_meta.get("task_type", "binary")
                inv_map = getattr(self.supervised_preprocessor, "inverse_label_mapping_", {})
                predicted_label = inv_map.get(pred, str(pred))

                is_attack = False
                if task_type == "binary" and pred == 1:
                    is_attack = True
                elif task_type != "binary" and str(predicted_label).upper() != "BENIGN":
                    is_attack = True

                if is_attack:
                    version = self.supervised_meta.get("version", "supervised_v1")
                    algo = self.supervised_meta.get("algorithm", "ML_Classifier")
                    conf_str = f" (Confidence: {confidence * 100:.1f}%)" if confidence else ""
                    alerts.append(
                        Alert(
                            timestamp=flow.end_time,
                            rule_name=f"ML_SUPERVISED_{predicted_label.upper()}",
                            src_ip=flow.src_ip,
                            dst_ip=flow.dst_ip,
                            observed_value=float(confidence if confidence else 1.0),
                            threshold=0.5,
                            window_seconds=round(flow.duration, 2),
                            severity="CRITICAL" if (confidence and confidence > 0.85) else "HIGH",
                            message=(
                                f"[ML SUPERVISED] {algo} classified flow as {predicted_label}{conf_str} "
                                f"over {flow.total_packets} packets ({flow.total_bytes} bytes)."
                            ),
                            detection_method="ML_SUPERVISED",
                            model_version=version,
                            confidence=confidence,
                            anomaly_score=None,
                        )
                    )
            except Exception as err:
                logger.debug("Supervised inference error on flow %s: %s", flow.flow_id, err)

        # B. Unsupervised Anomaly Detection (Isolation Forest)
        if self.enable_anomaly and self.anomaly_model is not None and self.anomaly_preprocessor is not None:
            try:
                X_vec = self.anomaly_preprocessor.transform_inference(features)
                # Compute raw anomaly decision and score
                raw_pred = self.anomaly_model.predict(X_vec)[0]
                raw_dec = self.anomaly_model.decision_function(X_vec)[0]
                # Normalized score: higher = more anomalous
                import math
                norm_score = 1.0 / (1.0 + math.exp(raw_dec * 5.0))

                if raw_pred == -1:  # Outlier
                    version = self.anomaly_meta.get("version", "anomaly_v1")
                    alerts.append(
                        Alert(
                            timestamp=flow.end_time,
                            rule_name="ML_ANOMALY_OUTLIER",
                            src_ip=flow.src_ip,
                            dst_ip=flow.dst_ip,
                            observed_value=round(norm_score, 4),
                            threshold=0.5,
                            window_seconds=round(flow.duration, 2),
                            severity="MEDIUM" if norm_score < 0.75 else "HIGH",
                            message=(
                                f"[ML ANOMALY] Isolation Forest identified statistical outlier (Score: {norm_score:.3f}) "
                                f"differing from benign baseline profile."
                            ),
                            detection_method="ML_ANOMALY",
                            model_version=version,
                            confidence=None,
                            anomaly_score=norm_score,
                        )
                    )
            except Exception as err:
                logger.debug("Anomaly inference error on flow %s: %s", flow.flow_id, err)

        return alerts
