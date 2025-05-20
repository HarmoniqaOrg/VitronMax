"""Type stubs for reportlab."""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# reportlab.lib.pagesizes
class pagesizes:
    letter: Tuple[float, float] = (612.0, 792.0)
    A4: Tuple[float, float] = (595.2755905511812, 841.8897637795277)

# reportlab.lib
class lib:
    class colors:
        black: Any
        white: Any
        red: Any
        green: Any
        blue: Any
        HexColor: Callable[[str], Any]
    
    class styles:
        @staticmethod
        def getSampleStyleSheet() -> Dict[str, Any]: ...
        class ParagraphStyle:
            def __init__(self, name: str, parent: Optional[Any] = None, **kw: Any) -> None: ...
    
    class units:
        inch: float = 72.0
        cm: float = 28.3464566929
    
    class enums:
        TA_LEFT: int = 0
        TA_CENTER: int = 1
        TA_RIGHT: int = 2
        TA_JUSTIFY: int = 4

# reportlab.platypus
class platypus:
    class SimpleDocTemplate:
        def __init__(
            self, 
            filename: str, 
            pagesize: Tuple[float, float] = ...,
            rightMargin: float = 72.0,
            leftMargin: float = 72.0,
            topMargin: float = 72.0,
            bottomMargin: float = 72.0,
            **kw: Any
        ) -> None: ...
        
        def build(self, flowables: List[Any]) -> None: ...
    
    class Paragraph:
        def __init__(self, text: str, style: Any, bulletText: Optional[str] = None) -> None: ...
    
    class Spacer:
        def __init__(self, width: float, height: float) -> None: ...
    
    class Table:
        def __init__(
            self, 
            data: List[List[Any]], 
            colWidths: Optional[List[float]] = None,
            rowHeights: Optional[List[float]] = None,
            style: Optional[Any] = None,
            splitByRow: int = 1,
            repeatRows: int = 0,
            repeatCols: int = 0,
            **kw: Any
        ) -> None: ...
    
    class TableStyle:
        def __init__(self, cmds: List[Tuple]) -> None: ...
    
    class Image:
        def __init__(
            self,
            filename: str,
            width: Optional[float] = None,
            height: Optional[float] = None,
            **kw: Any
        ) -> None: ...
    
    class flowables:
        class HRFlowable:
            def __init__(
                self,
                width: str = "100%",
                thickness: float = 1.0,
                color: Any = ...,
                **kw: Any
            ) -> None: ...
