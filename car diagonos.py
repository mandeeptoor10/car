from typing import List, Dict, Any, Optional

# --- Conceptual Type Definitions (Mocks for clarity) ---

class TelemetryData:
    """Represents a window of vehicle telemetry data."""
    def __init__(self, data: Dict[str, List[float]]):
        self._data = data

    def mean(self, parameter: str) -> Optional[float]:
        """Calculates the mean of a parameter in the telemetry window."""
        values = self._data.get(parameter, [])
        return sum(values) / len(values) if values else None

class Session:
    """Represents the current diagnostic session state."""
    def __init__(self, dtcs: List[str], meta: Dict[str, Any], telemetry_data: Dict[str, List[float]]):
        self.dtcs = dtcs
        self.meta = meta
        self._telemetry_data = telemetry_data

    def telemetry_window(self, duration_s: int = 60) -> TelemetryData:
        """Returns the last N seconds of telemetry data."""
        # In a real system, this would fetch the relevant slice of data
        return TelemetryData(self._telemetry_data)

# External Dependencies (Mocks)
class DTC_DB:
    @staticmethod
    def lookup(code: str) -> str:
        """Mocks a lookup against the DTC database."""
        return f"Database definition for {code}: Misfire detected."

class MLModel:
    @staticmethod
    def predict_rank(features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mocks the ML model prediction, returning ranked causes."""
        return [{"cause": "Spark Plug 2", "confidence": 0.9},
                {"cause": "O2 Sensor Bank 1", "confidence": 0.7}]

def build_features(telemetry: TelemetryData, dtcs: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
    """Mocks feature engineering for the ML model."""
    return {"dtc_count": len(dtcs), "age": meta.get("age", 5)}

def aggregate_findings(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mocks combining all findings into a final, structured report."""
    report = {"summary": "Analysis complete.", "details": findings}
    # In a real implementation, this would determine overall confidence.
    return report

# --- Constants for Symptom Checks ---
# Define threshold constants to make the rule logic clear and easy to modify.
FUEL_TRIM_HIGH_THRESHOLD = 15.0
O2_VOLTAGE_LOW_THRESHOLD = 0.2
MISFIRE_DTC_CYLINDER_2 = "P0302"


# --- Diagnostic Step Functions ---

def _run_dtc_mapping(dtcs: List[str]) -> List[Dict[str, Any]]:
    """Step 1: Perform rule-based DTC lookup and mapping."""
    dtc_findings: List[Dict[str, Any]] = []
    for code in dtcs:
        try:
            mapping = DTC_DB.lookup(code)
            dtc_findings.append({
                "source": "DTC_MAPPING",
                "code": code,
                "mapping": mapping,
                "confidence": 1.0
            })
        except Exception as e:
            print(f"Error looking up DTC {code}: {e}")
            # Continue processing other codes
            continue
    return dtc_findings


def _run_symptom_checks(dtcs: List[str], telemetry: TelemetryData) -> List[Dict[str, Any]]:
    """Step 2: Run specific, hard-coded symptom logic based on DTCs and telemetry."""
    symptom_findings: List[Dict[str, Any]] = []

    if MISFIRE_DTC_CYLINDER_2 in dtcs:
        # Check specific metrics relevant to a P0302 misfire
        ft = telemetry.mean("short_term_fuel_trim_bank1")
        o2 = telemetry.mean("o2_voltage_bank1")

        if ft is not None and o2 is not None and ft > FUEL_TRIM_HIGH_THRESHOLD and o2 < O2_VOLTAGE_LOW_THRESHOLD:
            symptom_findings.append({
                "source": "SYMPTOM_CHECK",
                "probable_cause": "injector_cylinder_2",
                "reason": f"High fuel trim ({ft:.1f}%) and low O2 voltage ({o2:.2f}V) suggest a lean condition at bank 1, common with an injector failure on that bank.",
                "confidence": 0.85
            })
    
    # Add other symptom checks here (e.g., overheating, low voltage, etc.)

    return symptom_findings


def _run_ml_prediction(session: Session, telemetry: TelemetryData) -> List[Dict[str, Any]]:
    """Step 3: Run the Machine Learning model prediction."""
    ml_findings: List[Dict[str, Any]] = []
    try:
        features = build_features(telemetry, session.dtcs, session.meta)
        ml_ranking = MLModel.predict_rank(features)

        # Reformat ML results to match the standardized findings structure
        for rank, result in enumerate(ml_ranking, 1):
             ml_findings.append({
                "source": "ML_MODEL",
                "rank": rank,
                "cause": result.get("cause"),
                "confidence": result.get("confidence", 0.0)
            })

    except Exception as e:
        print(f"ML Prediction failed: {e}")
        ml_findings.append({
            "source": "ML_MODEL",
            "error": "Prediction failed",
            "details": str(e),
            "confidence": 0.0
        })

    return ml_findings


# --- Main Diagnosis Orchestration ---

def diagnose(session: Session) -> Dict[str, Any]:
    """
    Performs a multi-faceted vehicle diagnosis by combining rule-based,
    symptom-based, and machine learning methods.

    Args:
        session: The current diagnostic session object containing DTCs and metadata.

    Returns:
        A structured report (Dict) summarizing all findings and confidence scores.
    """
    findings: List[Dict[str, Any]] = []

    # 1. Gather Telemetry data once for all checks
    telemetry = session.telemetry_window()

    # 2. Run Diagnostic Stages
    print("Running DTC Mapping...")
    findings += _run_dtc_mapping(session.dtcs)

    print("Running Symptom Checks...")
    findings += _run_symptom_checks(session.dtcs, telemetry)

    print("Running ML Prediction...")
    findings += _run_ml_prediction(session, telemetry)

    # 3. Final Composition
    print("Aggregating findings into final report...")
    report = aggregate_findings(findings)

    return report

# --- Example Usage (Demonstrates the flow) ---
if __name__ == '__main__':
    # Initialize a mock session with data that triggers the P0302 rule
    mock_session = Session(
        dtcs=["P0302", "U0100"],
        meta={"vehicle_model": "Viper", "age": 7, "mileage": 150000},
        telemetry_data={
            "short_term_fuel_trim_bank1": [16.5, 17.0, 15.5],  # High FT
            "o2_voltage_bank1": [0.15, 0.18, 0.19],            # Low O2
            "engine_temp": [90.0]
        }
    )

    final_report = diagnose(mock_session)
    import json
    print("\n--- Final Diagnostic Report ---")
    print(json.dumps(final_report, indent=4))
