"""
Utility Functions for Jupyter SQL Extension

This module provides helper functions for query validation, sanitization,
formatting, and other common operations used throughout the extension.
"""

import re
import time
import logging
import hashlib
import functools
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime
import sqlparse
from sqlparse.sql import Token, TokenList, Identifier, IdentifierList
from sqlparse.tokens import Keyword, Name, Punctuation, Whitespace

from .exceptions import ValidationError
from .types import QueryResult, OutputFormat, QueryMetadata


logger = logging.getLogger(__name__)


def sanitize_query(query: str) -> str:
    """
    Sanitize SQL query to prevent injection attacks.
    
    This function performs basic SQL injection prevention by:
    - Removing or escaping dangerous characters
    - Validating query structure
    - Limiting query complexity
    
    Args:
        query: Raw SQL query
        
    Returns:
        Sanitized SQL query
        
    Raises:
        ValidationError: If query contains dangerous content
    """
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty")
    
    # Remove leading/trailing whitespace
    query = query.strip()
    
    # Check for dangerous patterns
    dangerous_patterns = [
        r'--.*$',  # SQL comments
        r'/\*.*?\*/',  # Multi-line comments
        r'xp_',  # Extended stored procedures
        r'sp_',  # Stored procedures
        r'exec\s*\(',  # EXEC statements
        r'execute\s*\(',  # EXECUTE statements
        r'union\s+all\s+select',  # UNION ALL SELECT injection
        r'union\s+select',  # UNION SELECT injection
        r'waitfor\s+delay',  # WAITFOR DELAY
        r'benchmark\s*\(',  # BENCHMARK function
        r'sleep\s*\(',  # SLEEP function
    ]
    
    query_lower = query.lower()
    for pattern in dangerous_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE | re.MULTILINE):
            raise ValidationError(f"Query contains dangerous pattern: {pattern}")
    
    # Check for balanced parentheses
    if query.count('(') != query.count(')'):
        raise ValidationError("Unbalanced parentheses in query")
    
    # Check for balanced quotes
    single_quotes = query.count("'")
    double_quotes = query.count('"')
    if single_quotes % 2 != 0 or double_quotes % 2 != 0:
        raise ValidationError("Unbalanced quotes in query")
    
    # Limit query length
    if len(query) > 10000:  # 10KB limit
        raise ValidationError("Query too long (max 10KB)")
    
    return query


def validate_connection_id(connection_id: str) -> bool:
    """
    Validate connection ID format.
    
    Args:
        connection_id: Connection identifier to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not connection_id:
        return False
    
    # Connection ID should be alphanumeric with underscores and hyphens
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, connection_id))


def validate_sql_syntax(query: str) -> bool:
    """
    Validate SQL syntax using sqlparse.
    
    Args:
        query: SQL query to validate
        
    Returns:
        True if syntax is valid, False otherwise
    """
    try:
        # Parse the query
        parsed = sqlparse.parse(query)
        
        # Basic validation - check if parsing succeeded
        if not parsed:
            return False
        
        # Check for basic SQL structure
        statement = parsed[0]
        tokens = statement.tokens
        
        # Look for SQL keywords
        has_keyword = any(
            token.ttype == Keyword and token.value.upper() in [
                'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER', 
                'WITH', 'EXPLAIN', 'ANALYZE', 'DESC', 'SHOW', 'DESCRIBE', 'USE'
            ]
            for token in tokens
        )
        
        return has_keyword
        
    except Exception as e:
        logger.debug(f"SQL syntax validation failed: {e}")
        return False


def detect_query_type(query: str) -> str:
    """
    Detect the type of SQL query.
    
    Args:
        query: SQL query to analyze
        
    Returns:
        Query type ('select', 'insert', 'update', 'delete', 'create', 'drop', 'alter', 'other')
    """
    try:
        parsed = sqlparse.parse(query)
        if not parsed:
            return 'other'
        
        statement = parsed[0]
        tokens = statement.tokens
        
        # Find the first keyword
        for token in tokens:
            if token.ttype == Keyword:
                keyword = token.value.upper()
                if keyword in ['SELECT', 'WITH', 'EXPLAIN', 'ANALYZE', 'DESC', 'SHOW', 'DESCRIBE', 'USE']:
                    return 'select'
                elif keyword in ['INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
                    return 'insert'
                elif keyword in ['UPDATE']:
                    return 'update'
                elif keyword in ['DELETE', 'CREATE', 'DROP', 'ALTER']:
                    return 'delete'
        
        return 'other'
        
    except Exception as e:
        logger.debug(f"Query type detection failed: {e}")
        return 'other'


def extract_table_names(query: str) -> List[str]:
    """
    Extract table names from SQL query.
    
    Args:
        query: SQL query to analyze
        
    Returns:
        List of table names found in the query
    """
    try:
        parsed = sqlparse.parse(query)
        if not parsed:
            return []
        
        statement = parsed[0]
        tables = []
        
        # Extract table names from FROM and JOIN clauses
        for token in statement.tokens:
            if token.ttype == Keyword and token.value.upper() in ['FROM', 'JOIN']:
                # Look for identifiers after FROM/JOIN
                for i, t in enumerate(statement.tokens):
                    if t == token and i + 1 < len(statement.tokens):
                        next_token = statement.tokens[i + 1]
                        if isinstance(next_token, Identifier):
                            tables.append(str(next_token))
                        elif isinstance(next_token, Token) and next_token.ttype == Name:
                            tables.append(next_token.value)
        
        return list(set(tables))  # Remove duplicates
        
    except Exception as e:
        logger.debug(f"Table name extraction failed: {e}")
        return []


def format_query_result(
    result: QueryResult,
    output_format: OutputFormat,
    max_rows: Optional[int] = None
) -> Union[str, Dict[str, Any]]:
    """
    Format query result for display.
    
    Args:
        result: Query result to format
        output_format: Desired output format
        max_rows: Maximum number of rows to include
        
    Returns:
        Formatted result
    """
    if not result.data:
        return "No results"
    
    # Limit rows if specified
    data = result.data
    if max_rows and len(data) > max_rows:
        data = data[:max_rows]
    
    if output_format == OutputFormat.TABLE:
        return _format_as_table(data, result.columns)
    elif output_format == OutputFormat.JSON:
        return _format_as_json(data, result.columns, result.metadata)
    elif output_format == OutputFormat.CSV:
        return _format_as_csv(data, result.columns)
    elif output_format == OutputFormat.MARKDOWN:
        return _format_as_markdown(data, result.columns)
    else:
        return _format_as_table(data, result.columns)


def _format_as_table(data: List[List[Any]], columns: List[str]) -> str:
    """Format data as ASCII table."""
    if not data:
        return "No data"
    
    # Calculate column widths
    col_widths = []
    for i, col in enumerate(columns):
        max_width = len(str(col))
        for row in data:
            if i < len(row):
                max_width = max(max_width, len(str(row[i])))
        col_widths.append(max_width)
    
    # Build table
    lines = []
    
    # Header
    header = "| " + " | ".join(
        str(col).ljust(col_widths[i]) for i, col in enumerate(columns)
    ) + " |"
    lines.append(header)
    
    # Separator
    separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    lines.append(separator)
    
    # Data rows
    for row in data:
        formatted_row = "| " + " | ".join(
            str(row[i] if i < len(row) else "").ljust(col_widths[i])
            for i in range(len(columns))
        ) + " |"
        lines.append(formatted_row)
    
    return "\n".join(lines)


def _format_as_json(data: List[List[Any]], columns: List[str], metadata: QueryMetadata) -> Dict[str, Any]:
    """Format data as JSON."""
    return {
        "columns": columns,
        "data": data,
        "row_count": len(data),
        "metadata": metadata.to_dict()
    }


def _format_as_csv(data: List[List[Any]], columns: List[str]) -> str:
    """Format data as CSV."""
    if not data:
        return ""
    
    lines = []
    
    # Header
    lines.append(",".join(f'"{col}"' for col in columns))
    
    # Data rows
    for row in data:
        formatted_row = ",".join(
            f'"{str(row[i] if i < len(row) else "")}"'
            for i in range(len(columns))
        )
        lines.append(formatted_row)
    
    return "\n".join(lines)


def _format_as_markdown(data: List[List[Any]], columns: List[str]) -> str:
    """Format data as Markdown table."""
    if not data:
        return "No data"
    
    lines = []
    
    # Header
    header = "| " + " | ".join(str(col) for col in columns) + " |"
    lines.append(header)
    
    # Separator
    separator = "|" + "|".join(" --- " for _ in columns) + "|"
    lines.append(separator)
    
    # Data rows
    for row in data:
        formatted_row = "| " + " | ".join(
            str(row[i] if i < len(row) else "")
            for i in range(len(columns))
        ) + " |"
        lines.append(formatted_row)
    
    return "\n".join(lines)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        log_format: Log message format
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging to {log_file}: {e}")


def measure_performance(func: Callable) -> Callable:
    """
    Decorator to measure function performance.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function with performance measurement
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = _get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as e:
            success = False
            raise
        finally:
            end_time = time.time()
            end_memory = _get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory if end_memory and start_memory else None
            
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.3f}s "
                f"(memory: {memory_delta:+d} bytes, success: {success})"
            )
        
        return result
    
    return wrapper


def _get_memory_usage() -> Optional[int]:
    """Get current memory usage in bytes."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    except ImportError:
        return None


def generate_query_hash(query: str) -> str:
    """
    Generate a hash for a SQL query.
    
    Args:
        query: SQL query to hash
        
    Returns:
        Query hash string
    """
    # Normalize query (remove extra whitespace, convert to lowercase)
    normalized = re.sub(r'\s+', ' ', query.strip().lower())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate string to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated string
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def safe_str(value: Any) -> str:
    """
    Safely convert value to string.
    
    Args:
        value: Value to convert
        
    Returns:
        String representation
    """
    if value is None:
        return ""
    elif isinstance(value, (int, float, bool)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, datetime):
        return value.isoformat()
    else:
        return str(value)


def parse_connection_string(connection_string: str) -> Dict[str, str]:
    """
    Parse database connection string.
    
    Args:
        connection_string: Connection string to parse
        
    Returns:
        Dictionary of connection parameters
    """
    try:
        # Handle different connection string formats
        if connection_string.startswith('postgresql://'):
            return _parse_postgresql_connection(connection_string)
        elif connection_string.startswith('mysql://'):
            return _parse_mysql_connection(connection_string)
        elif connection_string.startswith('sqlite://'):
            return _parse_sqlite_connection(connection_string)
        else:
            # Generic parsing
            return _parse_generic_connection(connection_string)
    except Exception as e:
        raise ValidationError(f"Failed to parse connection string: {e}")


def _parse_postgresql_connection(connection_string: str) -> Dict[str, str]:
    """Parse PostgreSQL connection string."""
    # Remove protocol
    connection_string = connection_string.replace('postgresql://', '')
    
    # Split into parts
    if '@' in connection_string:
        auth_part, rest = connection_string.split('@', 1)
        if ':' in auth_part:
            username, password = auth_part.split(':', 1)
        else:
            username, password = auth_part, ""
    else:
        username, password = "", ""
        rest = connection_string
    
    # Parse host and database
    if '/' in rest:
        host_part, database = rest.split('/', 1)
        if ':' in host_part:
            host, port = host_part.split(':', 1)
        else:
            host, port = host_part, "5432"
    else:
        host, port = rest, "5432"
        database = ""
    
    return {
        "driver": "postgresql",
        "host": host,
        "port": port,
        "database": database,
        "username": username,
        "password": password
    }


def _parse_mysql_connection(connection_string: str) -> Dict[str, str]:
    """Parse MySQL connection string."""
    # Remove protocol
    connection_string = connection_string.replace('mysql://', '')
    
    # Split into parts
    if '@' in connection_string:
        auth_part, rest = connection_string.split('@', 1)
        if ':' in auth_part:
            username, password = auth_part.split(':', 1)
        else:
            username, password = auth_part, ""
    else:
        username, password = "", ""
        rest = connection_string
    
    # Parse host and database
    if '/' in rest:
        host_part, database = rest.split('/', 1)
        if ':' in host_part:
            host, port = host_part.split(':', 1)
        else:
            host, port = host_part, "3306"
    else:
        host, port = rest, "3306"
        database = ""
    
    return {
        "driver": "mysql",
        "host": host,
        "port": port,
        "database": database,
        "username": username,
        "password": password
    }


def _parse_sqlite_connection(connection_string: str) -> Dict[str, str]:
    """Parse SQLite connection string."""
    # Remove protocol
    database = connection_string.replace('sqlite://', '')
    
    return {
        "driver": "sqlite",
        "host": "",
        "port": "",
        "database": database,
        "username": "",
        "password": ""
    }


def _parse_generic_connection(connection_string: str) -> Dict[str, str]:
    """Parse generic connection string."""
    # Try to extract components using regex
    pattern = r'(\w+)://([^:@]+)(?::([^@]+))?@([^:/]+)(?::(\d+))?(?:/(.+))?'
    match = re.match(pattern, connection_string)
    
    if match:
        driver, username, password, host, port, database = match.groups()
        return {
            "driver": driver,
            "host": host,
            "port": port or "",
            "database": database or "",
            "username": username,
            "password": password or ""
        }
    else:
        raise ValidationError("Unable to parse connection string format")


def validate_connection_parameters(params: Dict[str, str]) -> bool:
    """
    Validate connection parameters.
    
    Args:
        params: Connection parameters to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["driver", "host"]
    
    for field in required_fields:
        if field not in params or not params[field]:
            return False
    
    # Validate port if present
    if "port" in params and params["port"]:
        try:
            port = int(params["port"])
            if not (1 <= port <= 65535):
                return False
        except ValueError:
            return False
    
    return True


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    if bytes_value == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    
    while bytes_value >= 1024 and unit_index < len(units) - 1:
        bytes_value /= 1024
        unit_index += 1
    
    return f"{bytes_value:.1f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "1m 30s")
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m" 