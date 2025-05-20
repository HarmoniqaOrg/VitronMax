"""Type stubs for joblib."""

from typing import Any, Optional, TypeVar, Callable

T = TypeVar('T')

def load(filename: str) -> Any: ...
def dump(obj: Any, filename: str, compress: int = 3, protocol: Optional[int] = None, cache_size: Optional[int] = None) -> None: ...

class Memory:
    def __init__(
        self, 
        location: Optional[str] = None, 
        backend: str = 'local',
        cachedir: Optional[str] = None,
        mmap_mode: Optional[str] = None,
        compress: bool = False,
        verbose: int = 0,
        bytes_limit: Optional[int] = None,
    ) -> None: ...
    
    def cache(self, func: Callable[..., T]) -> Callable[..., T]: ...
    def clear(self) -> None: ...
