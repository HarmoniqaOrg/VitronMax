"""
Blood-brain barrier permeability prediction module.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
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
        self.model: Any = joblib.load(str(model_path))
        self.version: str = MODEL_VERSION

    # ---------- public API -------------------------------------------------
    def predict(self, smiles: str) -> float:
        """Return BBB probability for a single SMILES (0.0 – 1.0)."""
        fp = self._featurise(smiles)  # raises ValueError if invalid
        # sklearn expects 2-D array for a single sample
        proba: float = float(self.model.predict_proba(fp[None, :])[0, 1])
        return proba

    # ---------- internal helpers ------------------------------------------
    @staticmethod
    def _featurise(smiles: str) -> npt.NDArray[np.int8]:
        """
        Convert SMILES → Morgan fingerprint (binary int8 vector).
        Raises ValueError on invalid SMILES.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")

        bitvect = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=FP_RADIUS, nBits=FP_BITS
        )
        # RDKit returns an ExplicitBitVect – convert to np.ndarray[int8]
        # The ToBitString() route is efficient.
        arr = np.zeros((FP_BITS,), dtype=np.int8)
        # Convert bit string to buffer, then check for '1' bytes
        on_bits = (
            np.frombuffer(bitvect.ToBitString().encode("utf-8"), dtype="S1") == b"1"
        )
        arr[on_bits] = 1
        return arr
