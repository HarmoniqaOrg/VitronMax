"""Type stubs for rdkit."""

from typing import Any, Optional

class Mol:
    """RDKit molecule representation."""

    def GetNumAtoms(self, onlyExplicit: bool = True) -> int: ...
    def GetNumBonds(self, onlyHeavy: bool = True) -> int: ...

class Chem:
    """RDKit chemistry utilities."""

    @staticmethod
    def MolFromSmiles(smiles: str, sanitize: bool = True) -> Optional[Mol]: ...
    @staticmethod
    def MolToSmiles(
        mol: Mol, isomericSmiles: bool = True, canonical: bool = True
    ) -> str: ...

    class AllChem:
        """RDKit all chemistry utilities."""

        @staticmethod
        def GetMorganFingerprintAsBitVect(
            mol: Mol,
            radius: int,
            nBits: int = 2048,
            useChirality: bool = False,
            useBondTypes: bool = True,
            useFeatures: bool = False,
        ) -> Any: ...
