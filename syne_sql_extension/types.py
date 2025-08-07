"""
Type Definitions for Jupyter SQL Extension

This module defines all the data structures, type aliases, and enums
used throughout the extension for type safety and consistency.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OutputFormat(str, Enum):
    """Available output formats for query results."""
    
    TABLE = "table"
    JSON = "json"
    DATAFRAME = "dataframe"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"


class ConnectionStatus(str, Enum):
    """Database connection status."""
    
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    UNKNOWN = "unknown"


class QueryType(str, Enum):
    """Types of SQL queries."""
    
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"
    OTHER = "other"


@dataclass
class QueryMetadata:
    """
    Metadata about a query execution.
    
    This contains information about query performance, timing,
    and execution details.
    """
    
    connection_id: str
    query_hash: int
    execution_time: float
    row_count: int
    success: bool
    timestamp: float
    
    # Optional fields
    query_type: Optional[QueryType] = None
    execution_plan: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    sql_state: Optional[str] = None
    error_code: Optional[int] = None
    
    # Performance metrics
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    network_bytes: Optional[int] = None
    
    # Additional context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    notebook_id: Optional[str] = None
    cell_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "connection_id": self.connection_id,
            "query_hash": self.query_hash,
            "execution_time": self.execution_time,
            "row_count": self.row_count,
            "success": self.success,
            "timestamp": self.timestamp,
            "query_type": self.query_type.value if self.query_type else None,
            "execution_plan": self.execution_plan,
            "error_message": self.error_message,
            "sql_state": self.sql_state,
            "error_code": self.error_code,
            "memory_usage": self.memory_usage,
            "cpu_usage": self.cpu_usage,
            "network_bytes": self.network_bytes,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "notebook_id": self.notebook_id,
            "cell_id": self.cell_id,
        }


@dataclass
class QueryResult:
    """
    Result of a SQL query execution.
    
    This contains the actual data returned by the query along with
    metadata about the execution.
    """
    
    data: List[List[Any]]
    columns: List[str]
    row_count: int
    metadata: QueryMetadata
    
    # Optional fields
    total_rows: Optional[int] = None  # For paginated results
    has_more: bool = False  # Whether there are more rows available
    
    def __post_init__(self):
        """Validate query result after initialization."""
        if not isinstance(self.data, list):
            raise ValueError("Data must be a list of rows")
        if not isinstance(self.columns, list):
            raise ValueError("Columns must be a list")
        if len(self.columns) == 0 and len(self.data) > 0:
            raise ValueError("Cannot have data without columns")
        
        # Validate that all rows have the same number of columns
        if self.data:
            expected_cols = len(self.columns)
            for i, row in enumerate(self.data):
                if len(row) != expected_cols:
                    raise ValueError(f"Row {i} has {len(row)} columns, expected {expected_cols}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "data": self.data,
            "columns": self.columns,
            "row_count": self.row_count,
            "metadata": self.metadata.to_dict(),
            "total_rows": self.total_rows,
            "has_more": self.has_more,
        }
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.data, columns=self.columns)
        except ImportError:
            raise ImportError("pandas is required to convert to DataFrame")
    
    def get_column_types(self) -> Dict[str, str]:
        """Infer column types from the data."""
        if not self.data:
            return {col: "unknown" for col in self.columns}
        
        types = {}
        for i, col in enumerate(self.columns):
            sample_values = [row[i] for row in self.data if row[i] is not None]
            if not sample_values:
                types[col] = "unknown"
                continue
            
            # Simple type inference
            sample = sample_values[0]
            if isinstance(sample, bool):
                types[col] = "boolean"
            elif isinstance(sample, int):
                types[col] = "integer"
            elif isinstance(sample, float):
                types[col] = "float"
            elif isinstance(sample, str):
                types[col] = "string"
            elif isinstance(sample, datetime):
                types[col] = "datetime"
            else:
                types[col] = "unknown"
        
        return types


@dataclass
class ConnectionInfo:
    """
    Information about a database connection.
    
    This contains connection details and status information.
    """
    
    connection_id: str
    name: str
    host: str
    port: int
    database: str
    driver: str
    status: ConnectionStatus
    
    # Optional fields
    username: Optional[str] = None
    ssl_mode: Optional[str] = None
    connection_timeout: Optional[int] = None
    query_timeout: Optional[int] = None
    
    # Status information
    last_connected: Optional[datetime] = None
    last_error: Optional[str] = None
    active_connections: int = 0
    max_connections: Optional[int] = None
    
    # Metadata
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "connection_id": self.connection_id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "driver": self.driver,
            "status": self.status.value,
            "username": self.username,
            "ssl_mode": self.ssl_mode,
            "connection_timeout": self.connection_timeout,
            "query_timeout": self.query_timeout,
            "last_connected": self.last_connected.isoformat() if self.last_connected else None,
            "last_error": self.last_error,
            "active_connections": self.active_connections,
            "max_connections": self.max_connections,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "description": self.description,
            "tags": self.tags,
        }


@dataclass
class QueryOptions:
    """
    Options for query execution.
    
    This contains various parameters that control how queries are executed.
    """
    
    timeout: Optional[int] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    explain: bool = False
    dry_run: bool = False
    
    # Output options
    output_format: OutputFormat = OutputFormat.TABLE
    include_metadata: bool = True
    max_display_rows: Optional[int] = None
    
    # Performance options
    cache_results: bool = True
    cache_ttl: Optional[int] = None  # Time to live in seconds
    
    # Security options
    allow_write_operations: bool = False
    validate_sql: bool = True
    sanitize_input: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timeout": self.timeout,
            "limit": self.limit,
            "offset": self.offset,
            "explain": self.explain,
            "dry_run": self.dry_run,
            "output_format": self.output_format.value,
            "include_metadata": self.include_metadata,
            "max_display_rows": self.max_display_rows,
            "cache_results": self.cache_results,
            "cache_ttl": self.cache_ttl,
            "allow_write_operations": self.allow_write_operations,
            "validate_sql": self.validate_sql,
            "sanitize_input": self.sanitize_input,
        }


@dataclass
class ExtensionConfig:
    """
    Configuration for the Jupyter SQL Extension.
    
    This contains all the settings that control the behavior of the extension.
    """
    
    # Service configuration
    sql_service_url: str = "http://localhost:8080/api/v1"
    sql_service_api_key: Optional[str] = None
    
    # Connection settings
    default_timeout: int = 30
    max_retries: int = 3
    connection_pool_size: int = 10
    
    # Query settings
    query_timeout: int = 300
    max_query_length: int = 10000
    allow_write_operations: bool = False
    allow_non_select_queries: bool = False
    
    # Output settings
    default_output_format: OutputFormat = OutputFormat.TABLE
    max_display_rows: int = 1000
    enable_rich_output: bool = True
    
    # Security settings
    enable_sql_validation: bool = True
    enable_input_sanitization: bool = True
    allowed_sql_keywords: List[str] = field(default_factory=lambda: [
        "SELECT", "FROM", "WHERE", "ORDER BY", "GROUP BY", "HAVING",
        "LIMIT", "OFFSET", "JOIN", "LEFT JOIN", "RIGHT JOIN", "INNER JOIN",
        "UNION", "UNION ALL", "DISTINCT", "AS", "COUNT", "SUM", "AVG", "MAX", "MIN"
    ])
    
    # Logging settings
    enable_logging: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Caching settings
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 1000
    
    # Performance settings
    enable_metrics: bool = True
    enable_profiling: bool = False
    max_concurrent_queries: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "sql_service_url": self.sql_service_url,
            "sql_service_api_key": "***" if self.sql_service_api_key else None,
            "default_timeout": self.default_timeout,
            "max_retries": self.max_retries,
            "connection_pool_size": self.connection_pool_size,
            "query_timeout": self.query_timeout,
            "max_query_length": self.max_query_length,
            "allow_write_operations": self.allow_write_operations,
            "allow_non_select_queries": self.allow_non_select_queries,
            "default_output_format": self.default_output_format.value,
            "max_display_rows": self.max_display_rows,
            "enable_rich_output": self.enable_rich_output,
            "enable_sql_validation": self.enable_sql_validation,
            "enable_input_sanitization": self.enable_input_sanitization,
            "allowed_sql_keywords": self.allowed_sql_keywords,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "log_format": self.log_format,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "max_cache_size": self.max_cache_size,
            "enable_metrics": self.enable_metrics,
            "enable_profiling": self.enable_profiling,
            "max_concurrent_queries": self.max_concurrent_queries,
        }


# Type aliases for common patterns
JSONValue = Union[str, int, float, bool, None, List[Any], Dict[str, Any]]
SQLQuery = str
ConnectionID = str
UserID = str
SessionID = str

# Result types for different operations
QueryResultType = Union[QueryResult, Dict[str, Any], str]
ValidationResult = Dict[str, Any]
ConnectionStatusResult = Dict[str, Any]

# Configuration types
ConfigDict = Dict[str, Any]
EnvironmentDict = Dict[str, str]

# Error types
ErrorDetails = Dict[str, Any]
ErrorResponse = Dict[str, Union[str, ErrorDetails]]

# Cache types
CacheKey = str
CacheValue = Any
CacheEntry = Dict[str, Union[CacheValue, float]]  # value and timestamp

# Metrics types
MetricValue = Union[int, float, str]
MetricTags = Dict[str, str]
MetricData = Dict[str, MetricValue] 