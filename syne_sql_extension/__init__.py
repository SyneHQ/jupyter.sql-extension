"""
Jupyter SQL Extension

A robust Jupyter Notebook extension for executing SQL queries through
internal service connections with security, validation, and rich output formatting.

This extension provides:
- %%sql cell magic for secure SQL execution
- Connection management with credential retrieval
- Rich output formatting (DataFrames, tables, JSON)
- Comprehensive error handling and validation
- Type-safe implementation with full documentation
"""

from typing import Dict, List, Optional, Any, Union
import sys
import warnings

# Package metadata
__version__ = "1.0.9"
__author__ = "SyneHQ Team"
__email__ = "dev@synehq.com"
__license__ = "MIT"
__description__ = "Enterprise-grade Jupyter extension for secure SQL query execution"

# Minimum Python version check
MIN_PYTHON_VERSION = (3, 8)
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"This package requires Python {'.'.join(map(str, MIN_PYTHON_VERSION))} or higher. "
        f"You are running Python {'.'.join(map(str, sys.version_info[:2]))}."
    )

# Import core components
try:
    from .magic import SQLConnectMagic, register_magic
    from .client import SQLServiceClient, ConnectionConfig
    from .exceptions import (
        SQLExtensionError,
        ConnectionError,
        AuthenticationError,
        ValidationError,
        QueryExecutionError,
        ConfigurationError
    )
    from .config import ExtensionConfig, load_config, validate_config
    from .utils import (
        format_query_result,
        sanitize_query,
        validate_connection_id,
        validate_sql_syntax,
        detect_query_type,
        extract_table_names,
        setup_logging,
        measure_performance
    )
    from .types import (
        QueryResult,
        ConnectionInfo,
        QueryMetadata,
        OutputFormat
    )

    # Import optional components
    _OPTIONAL_IMPORTS = {}

    try:
        from .grpc_client import GRPCServiceClient
        _OPTIONAL_IMPORTS['grpc'] = True
    except ImportError:
        _OPTIONAL_IMPORTS['grpc'] = False

    try:
        from .monitoring import MetricsCollector, PerformanceTracker
        _OPTIONAL_IMPORTS['monitoring'] = True
    except ImportError:
        _OPTIONAL_IMPORTS['monitoring'] = False

    try:
        from .encryption import SecureCredentialManager
        _OPTIONAL_IMPORTS['encryption'] = True
    except ImportError:
        _OPTIONAL_IMPORTS['encryption'] = False

except ImportError as e:
    # Graceful degradation for missing dependencies
    warnings.warn(
        f"Failed to import some components of jupyter_sql_extension: {e}. "
        "Some functionality may be limited. Please check your installation.",
        ImportWarning,
        stacklevel=2
    )

    # Define minimal fallback components
    class SQLConnectMagic:
        """Fallback magic class"""
        pass

    def register_magic():
        """Fallback registration function"""
        warnings.warn("Magic registration failed due to import errors", UserWarning)
        return False
    
    # Initialize empty optional imports for fallback
    _OPTIONAL_IMPORTS = {
        'grpc': False,
        'monitoring': False,
        'encryption': False
    }

# Extension metadata for Jupyter
_jupyter_extension_metadata = {
    "name": "jupyter_sql_extension",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "requires": {
        "notebook": ">=6.4.0",
        "ipython": ">=8.0.0",
    },
    "optional_features": _OPTIONAL_IMPORTS,
}

# Public API exports
__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",

    # Core components
    "SQLConnectMagic",
    "register_magic",
    "SQLServiceClient",
    "ConnectionConfig",
    "ExtensionConfig",

    # Configuration
    "load_config",
    "validate_config",
    "setup_logging",

    # Exceptions
    "SQLExtensionError",
    "ConnectionError",
    "AuthenticationError",
    "ValidationError",
    "QueryExecutionError",
    "ConfigurationError",

    # Types
    "QueryResult",
    "ConnectionInfo",
    "QueryMetadata",
    "OutputFormat",

    # Utilities
    "format_query_result",
    "sanitize_query",
    "validate_connection_id",
    "validate_sql_syntax",
    "detect_query_type",
    "extract_table_names",
    "setup_logging",
    "measure_performance",

    # Extension metadata
    "get_extension_info",
    "get_optional_features",
]

# Optional exports (only available if dependencies are installed)
if _OPTIONAL_IMPORTS.get('grpc', False):
    __all__.extend(["GRPCServiceClient"])

if _OPTIONAL_IMPORTS.get('monitoring', False):
    __all__.extend(["MetricsCollector", "PerformanceTracker"])

if _OPTIONAL_IMPORTS.get('encryption', False):
    __all__.extend(["SecureCredentialManager"])


def get_extension_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the extension.

    Returns:
        Dict containing extension metadata, version, capabilities, etc.
    """
    return {
        "name": "jupyter_sql_extension",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "license": __license__,
        "homepage": "https://github.com/diagonal/jupyter-sql-extension",
        "capabilities": {
            "sql_magic": True,
            "connection_management": True,
            "rich_output": True,
            "error_handling": True,
            "type_safety": True,
            "async_support": True,
            "grpc_support": _OPTIONAL_IMPORTS.get('grpc', False),
            "monitoring": _OPTIONAL_IMPORTS.get('monitoring', False),
            "encryption": _OPTIONAL_IMPORTS.get('encryption', False),
        },
        "python_version": {
            "minimum": ".".join(map(str, MIN_PYTHON_VERSION)),
            "current": ".".join(map(str, sys.version_info[:3])),
        },
        "jupyter_metadata": _jupyter_extension_metadata,
    }


def get_optional_features() -> Dict[str, bool]:
    """
    Get status of optional features based on available dependencies.

    Returns:
        Dict mapping feature names to availability status.
    """
    return _OPTIONAL_IMPORTS.copy()


def _check_jupyter_environment() -> bool:
    """
    Check if we're running in a Jupyter environment.

    Returns:
        True if running in Jupyter, False otherwise.
    """
    try:
        # Check for IPython
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is None:
            return False

        # Check for Jupyter-specific attributes
        return hasattr(ipython, 'kernel') or 'jupyter' in str(type(ipython))
    except ImportError:
        return False


def _auto_register_magic() -> bool:
    """
    Automatically register the SQL magic if in Jupyter environment.

    Returns:
        True if registration succeeded, False otherwise.
    """
    if not _check_jupyter_environment():
        return False

    try:
        return register_magic()
    except Exception as e:
        warnings.warn(
            f"Failed to auto-register SQL magic: {e}. "
            "You may need to call register_magic() manually.",
            UserWarning,
            stacklevel=2
        )
        return False


# Auto-register magic if we're in a Jupyter environment
if _check_jupyter_environment():
    _auto_register_magic()


# Compatibility aliases for backward compatibility
# (in case users were importing from older versions)
try:
    SqlConnectMagic = SQLConnectMagic  # Legacy name
    sql_service_client = SQLServiceClient  # Legacy snake_case import
except NameError:
    # Handle case where imports failed
    SqlConnectMagic = SQLConnectMagic  # Will be the fallback class
    sql_service_client = None


def load_ipython_extension(ipython):
    """
    Load the extension in IPython/Jupyter.

    This function is called when the extension is loaded via:
    %load_ext jupyter_sql_extension

    Args:
        ipython: The IPython instance
    """
    try:
        if register_magic():
            print("✅ Jupyter SQL Extension loaded successfully!")
            print(f"   Version: {__version__}")
            print("   Use %%sql to execute SQL queries.")

            # Show optional features status
            optional_features = get_optional_features()
            enabled_features = [name for name, enabled in optional_features.items() if enabled]
            if enabled_features:
                print(f"   Optional features enabled: {', '.join(enabled_features)}")
        else:
            print("⚠️  Jupyter SQL Extension loaded with limited functionality.")
            print("   Some features may not be available due to missing dependencies.")
    except Exception as e:
        print(f"❌ Failed to load Jupyter SQL Extension: {e}")
        raise


def unload_ipython_extension(ipython):
    """
    Unload the extension in IPython/Jupyter.

    This function is called when the extension is unloaded via:
    %unload_ext jupyter_sql_extension

    Args:
        ipython: The IPython instance
    """
    try:
        # Remove the magic command
        if hasattr(ipython, 'magics_manager'):
            magic_name = 'sql'
            if magic_name in ipython.magics_manager.magics['cell']:
                del ipython.magics_manager.magics['cell'][magic_name]

        print("✅ Jupyter SQL Extension unloaded successfully!")
    except Exception as e:
        print(f"⚠️  Error during extension unload: {e}")


# Package initialization message (only shown in development mode)
if __debug__ and _check_jupyter_environment():
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Jupyter SQL Extension v{__version__} initialized")
    logger.debug(f"Optional features: {get_optional_features()}")
