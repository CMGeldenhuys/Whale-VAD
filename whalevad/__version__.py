import importlib.metadata

try:
    __version__ = importlib.metadata.version("whalevad")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
