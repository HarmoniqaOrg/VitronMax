"""
Report generation functionality for the VitronMax API.
Provides PDF report generation capabilities.
"""
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,

)
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus.flowables import HRFlowable

from app.predict import BBBPredictor


LOGO_PATH = Path(__file__).parent / "static" / "vitronmax_logo.png"
DEFAULT_LOGO_PATH = Path(__file__).parent / "static" / "default_logo.png"


class PDFReportGenerator:
    """
    Generates PDF reports for molecule predictions.
    """

    def __init__(self, smiles: str):
        """
        Initialize the PDF report generator.

        Args:
            smiles: SMILES string of the molecule
        """
        self.smiles = smiles
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.buffer = BytesIO()
        self.prediction_data: Dict[str, Any] = {}
        self.doc = SimpleDocTemplate(
            self.buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        self.styles = getSampleStyleSheet()
        self._configure_styles()

    def _configure_styles(self) -> None:
        """Configure custom styles for the PDF document."""
        # Check if styles already exist before adding them
        if "ReportTitle" not in self.styles.byName:
            self.styles.add(
                ParagraphStyle(
                    name="ReportTitle",
                    parent=self.styles["Heading1"],
                    alignment=TA_CENTER,
                    fontSize=20,
                )
            )
        
        if "ReportSubtitle" not in self.styles.byName:
            self.styles.add(
                ParagraphStyle(
                    name="ReportSubtitle",
                    parent=self.styles["Heading2"],
                    fontSize=14,
                )
            )
            
        if "ReportTableHeader" not in self.styles.byName:
            self.styles.add(
                ParagraphStyle(
                    name="ReportTableHeader",
                    parent=self.styles["Normal"],
                    fontName="Helvetica-Bold",
                    fontSize=10,
                )
            )
            
        if "ReportFooter" not in self.styles.byName:
            self.styles.add(
                ParagraphStyle(
                    name="ReportFooter",
                    parent=self.styles["Normal"],
                    fontSize=8,
                    alignment=TA_CENTER,
                )
            )

    def generate_prediction_data(self) -> None:
        """Generate prediction data for the report."""
        # Create BBB predictor
        predictor = BBBPredictor()
        
        # Attempt prediction (which validates SMILES internally)
        try:
            bbb_prob = predictor.predict(self.smiles)
        except ValueError:
            raise ValueError(f"Invalid SMILES string: {self.smiles}")

        # Store prediction data
        self.prediction_data = {
            "smiles": self.smiles,
            "bbb_probability": bbb_prob,
            "model_version": predictor.version,
            "timestamp": self.timestamp,
            "interpretation": self._get_interpretation(bbb_prob),
        }

    def _get_interpretation(self, probability: float) -> str:
        """
        Get interpretation text based on BBB probability.

        Args:
            probability: Predicted BBB probability

        Returns:
            str: Interpretation text
        """
        if probability >= 0.9:
            return "Very likely to cross the blood-brain barrier (BBB)"
        elif probability >= 0.7:
            return "Likely to cross the blood-brain barrier (BBB)"
        elif probability >= 0.3:
            return "Uncertain BBB permeability"
        elif probability >= 0.1:
            return "Unlikely to cross the blood-brain barrier (BBB)"
        else:
            return "Very unlikely to cross the blood-brain barrier (BBB)"

    def _create_header(self) -> List[Any]:
        """
        Create the header section of the report.

        Returns:
            List: List of elements for the header
        """
        elements: List[Any] = []

        # Skip logo for now - we'll add a proper logo in production
        # Logo handling is removed from tests to avoid image format issues
        
        # Add title
        elements.append(Paragraph("VitronMax BBB Permeability Report", self.styles["ReportTitle"]))
        elements.append(Spacer(1, 0.25*inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        elements.append(Spacer(1, 0.25*inch))
        return elements

    def _create_molecule_section(self) -> List[Any]:
        """
        Create the molecule section of the report.

        Returns:
            List: List of elements for the molecule section
        """
        elements: List[Any] = []
        elements.append(Paragraph("Molecule Information", self.styles["ReportSubtitle"]))
        elements.append(Spacer(1, 0.1*inch))

        # Create a table for molecule information
        data = [
            ["SMILES", self.prediction_data["smiles"]],
        ]
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.black),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (1, 0), (-1, -1), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        return elements

    def _create_prediction_section(self) -> List[Any]:
        """
        Create the prediction section of the report.

        Returns:
            List: List of elements for the prediction section
        """
        elements: List[Any] = []
        elements.append(Paragraph("Prediction Results", self.styles["ReportSubtitle"]))
        elements.append(Spacer(1, 0.1*inch))

        # Create a table for prediction results
        bbb_prob = self.prediction_data["bbb_probability"]
        bbb_prob_str = f"{bbb_prob:.2f} ({self.prediction_data['interpretation']})"
        
        data = [
            ["BBB Permeability Probability", bbb_prob_str],
            ["Model Version", self.prediction_data["model_version"]],
            ["Prediction Date", self.prediction_data["timestamp"]],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.black),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (1, 0), (-1, -1), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.2*inch))
        return elements

    def _create_footer(self) -> List[Any]:
        """
        Create the footer section of the report.

        Returns:
            List: List of elements for the footer
        """
        elements: List[Any] = []
        elements.append(Spacer(1, 0.5*inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        elements.append(Spacer(1, 0.1*inch))
        
        footer_text = (
            "This report is generated by VitronMax API. "
            "Predictions are based on machine learning models and should be "
            "used for research purposes only. "
            f"Generated on {self.timestamp}"
        )
        elements.append(Paragraph(footer_text, self.styles["ReportFooter"]))
        return elements

    def generate_report(self) -> BytesIO:
        """
        Generate the PDF report.

        Returns:
            BytesIO: PDF report as a BytesIO object
        """
        # Generate prediction data
        self.generate_prediction_data()
        
        # Create the document content
        elements = []
        
        # Add header
        elements.extend(self._create_header())
        
        # Add molecule section
        elements.extend(self._create_molecule_section())
        
        # Add prediction section
        elements.extend(self._create_prediction_section())
        
        # Add footer
        elements.extend(self._create_footer())
        
        # Build the document
        self.doc.build(elements)
        
        # Reset buffer position to start
        self.buffer.seek(0)
        return self.buffer


def generate_pdf_report(smiles: str) -> BytesIO:
    """
    Generate a PDF report for a molecule.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        BytesIO: PDF report as a BytesIO object
    """
    generator = PDFReportGenerator(smiles)
    return generator.generate_report()
