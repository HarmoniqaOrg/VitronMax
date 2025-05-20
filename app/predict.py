"""
Blood-brain barrier permeability prediction module.
"""

from pathlib import Path
from typing import Any, List

import joblib
from rdkit import Chem
from rdkit.Chem import AllChem

# ──────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "models" / "bbb_rf_v1_0.joblib"
MODEL_VERSION = "1.0"
FP_BITS = 2048
FP_RADIUS = 2
# ──────────────────────────────────────────


class BBBPredictor:
    """Random-Forest BBB permeability predictor (Morgan FP 2–2048)."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model: Any = joblib.load(model_path)
        self.version: str = MODEL_VERSION

    # ---------- public API -------------------------------------------------
    def predict(self, smiles: str) -> float:
        """Return BBB probability for a single SMILES (0.0 – 1.0)."""
        fp = self._featurise(smiles)  # raises ValueError if invalid
        proba: float = float(self.model.predict_proba([fp])[0, 1])
        return proba

    # ---------- internal helpers ------------------------------------------
    @staticmethod
    def _featurise(smiles: str) -> List[int]:
        """Convert SMILES ➜ 2048-bit Morgan fingerprint."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        bitvect = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=FP_RADIUS, nBits=FP_BITS
        )
        # FastAPI JSON encoder handles NumPy int* poorly – cast to int list
        return list(bitvect)
