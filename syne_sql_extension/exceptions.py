"""
Custom Exceptions for Jupyter SQL Extension

This module defines all custom exceptions used throughout the extension,
providing clear error categorization and meaningful error messages.
"""

from typing import Optional, Dict, Any


class SQLExtensionError(Exception):
    """
    Base exception for all SQL extension errors.
    
    This is the parent class for all custom exceptions in the extension.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the base exception.
        
        Args:
            message: Human-readable error message
            details: Optional additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(SQLExtensionError):
    """
    Exception raised when configuration is invalid or missing.
    
    This exception is raised when:
    - Required configuration files are missing
    - Configuration values are invalid
    - Environment variables are not set
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None, config_value: Optional[Any] = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Optional configuration key that caused the error
            config_value: Optional configuration value that caused the error
        """
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)
        
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class ValidationError(SQLExtensionError):
    """
    Exception raised when input validation fails.
    
    This exception is raised when:
    - SQL queries fail validation
    - Connection IDs are invalid
    - Parameters are malformed
    """
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Optional field that failed validation
            value: Optional value that failed validation
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(message, details)
        self.field = field
        self.value = value


class AuthenticationError(SQLExtensionError):
    """
    Exception raised when authentication fails.
    
    This exception is raised when:
    - API keys are invalid or expired
    - User credentials are incorrect
    - Authentication tokens are missing
    """
    
    def __init__(self, message: str, auth_method: Optional[str] = None, retry_after: Optional[int] = None):
        """
        Initialize authentication error.
        
        Args:
            message: Error message
            auth_method: Optional authentication method that failed
            retry_after: Optional seconds to wait before retrying
        """
        details = {}
        if auth_method:
            details["auth_method"] = auth_method
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(message, details)
        self.auth_method = auth_method
        self.retry_after = retry_after


class ConnectionError(SQLExtensionError):
    """
    Exception raised when database connection fails.
    
    This exception is raised when:
    - Database server is unreachable
    - Connection configuration is invalid
    - Network issues prevent connection
    """
    
    def __init__(self, message: str, connection_id: Optional[str] = None, host: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize connection error.
        
        Args:
            message: Error message
            connection_id: Optional connection identifier
            host: Optional database host
            port: Optional database port
        """
        details = {}
        if connection_id:
            details["connection_id"] = connection_id
        if host:
            details["host"] = host
        if port:
            details["port"] = port
        
        super().__init__(message, details)
        self.connection_id = connection_id
        self.host = host
        self.port = port


class QueryExecutionError(SQLExtensionError):
    """
    Exception raised when SQL query execution fails.
    
    This exception is raised when:
    - SQL syntax is invalid
    - Query times out
    - Database returns an error
    - Insufficient permissions
    """
    
    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        sql_state: Optional[str] = None,
        error_code: Optional[int] = None,
        execution_time: Optional[float] = None
    ):
        """
        Initialize query execution error.
        
        Args:
            message: Error message
            query: Optional SQL query that failed
            sql_state: Optional SQL state code
            error_code: Optional database error code
            execution_time: Optional time spent before error
        """
        details = {}
        if query:
            details["query"] = query
        if sql_state:
            details["sql_state"] = sql_state
        if error_code:
            details["error_code"] = error_code
        if execution_time:
            details["execution_time"] = execution_time
        
        super().__init__(message, details)
        self.query = query
        self.sql_state = sql_state
        self.error_code = error_code
        self.execution_time = execution_time


class TimeoutError(SQLExtensionError):
    """
    Exception raised when operations timeout.
    
    This exception is raised when:
    - Query execution exceeds timeout
    - Connection establishment times out
    - API requests timeout
    """
    
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, operation: Optional[str] = None):
        """
        Initialize timeout error.
        
        Args:
            message: Error message
            timeout_seconds: Optional timeout duration in seconds
            operation: Optional operation that timed out
        """
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation
        
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class RateLimitError(SQLExtensionError):
    """
    Exception raised when rate limits are exceeded.
    
    This exception is raised when:
    - Too many requests are made in a time period
    - API rate limits are hit
    - Database connection limits are exceeded
    """
    
    def __init__(self, message: str, retry_after: Optional[int] = None, limit_type: Optional[str] = None):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            retry_after: Optional seconds to wait before retrying
            limit_type: Optional type of rate limit (e.g., 'requests', 'connections')
        """
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        if limit_type:
            details["limit_type"] = limit_type
        
        super().__init__(message, details)
        self.retry_after = retry_after
        self.limit_type = limit_type


class DataProcessingError(SQLExtensionError):
    """
    Exception raised when data processing fails.
    
    This exception is raised when:
    - Result formatting fails
    - Data type conversion fails
    - Output generation fails
    """
    
    def __init__(self, message: str, data_type: Optional[str] = None, processing_step: Optional[str] = None):
        """
        Initialize data processing error.
        
        Args:
            message: Error message
            data_type: Optional type of data being processed
            processing_step: Optional step where processing failed
        """
        details = {}
        if data_type:
            details["data_type"] = data_type
        if processing_step:
            details["processing_step"] = processing_step
        
        super().__init__(message, details)
        self.data_type = data_type
        self.processing_step = processing_step


# Exception mapping for error handling
EXCEPTION_MAPPING = {
    "ConfigurationError": ConfigurationError,
    "ValidationError": ValidationError,
    "AuthenticationError": AuthenticationError,
    "ConnectionError": ConnectionError,
    "QueryExecutionError": QueryExecutionError,
    "TimeoutError": TimeoutError,
    "RateLimitError": RateLimitError,
    "DataProcessingError": DataProcessingError,
}


def create_exception_from_dict(error_data: Dict[str, Any]) -> SQLExtensionError:
    """
    Create an exception instance from dictionary data.
    
    Args:
        error_data: Dictionary containing error information
        
    Returns:
        Appropriate exception instance
        
    Raises:
        ValueError: If error data is invalid
    """
    error_type = error_data.get("error_type")
    message = error_data.get("message", "Unknown error")
    details = error_data.get("details", {})
    
    if error_type not in EXCEPTION_MAPPING:
        # Fallback to base exception
        return SQLExtensionError(message, details)
    
    exception_class = EXCEPTION_MAPPING[error_type]
    
    # Create exception with appropriate parameters based on type
    if error_type == "ConfigurationError":
        return exception_class(
            message,
            config_key=details.get("config_key"),
            config_value=details.get("config_value")
        )
    elif error_type == "ValidationError":
        return exception_class(
            message,
            field=details.get("field"),
            value=details.get("value")
        )
    elif error_type == "AuthenticationError":
        return exception_class(
            message,
            auth_method=details.get("auth_method"),
            retry_after=details.get("retry_after")
        )
    elif error_type == "ConnectionError":
        return exception_class(
            message,
            connection_id=details.get("connection_id"),
            host=details.get("host"),
            port=details.get("port")
        )
    elif error_type == "QueryExecutionError":
        return exception_class(
            message,
            query=details.get("query"),
            sql_state=details.get("sql_state"),
            error_code=details.get("error_code"),
            execution_time=details.get("execution_time")
        )
    elif error_type == "TimeoutError":
        return exception_class(
            message,
            timeout_seconds=details.get("timeout_seconds"),
            operation=details.get("operation")
        )
    elif error_type == "RateLimitError":
        return exception_class(
            message,
            retry_after=details.get("retry_after"),
            limit_type=details.get("limit_type")
        )
    elif error_type == "DataProcessingError":
        return exception_class(
            message,
            data_type=details.get("data_type"),
            processing_step=details.get("processing_step")
        )
    else:
        return exception_class(message, details)


def is_retryable_error(exception: Exception) -> bool:
    """
    Check if an exception is retryable.
    
    Args:
        exception: Exception to check
        
    Returns:
        True if the exception is retryable, False otherwise
    """
    retryable_exceptions = (
        ConnectionError,
        TimeoutError,
        RateLimitError
    )
    
    return isinstance(exception, retryable_exceptions)


def get_retry_delay(exception: Exception, attempt: int) -> float:
    """
    Calculate retry delay for an exception.
    
    Args:
        exception: Exception that occurred
        attempt: Current attempt number (1-based)
        
    Returns:
        Delay in seconds before next retry
    """
    base_delay = 1.0
    
    if isinstance(exception, RateLimitError) and exception.retry_after:
        return float(exception.retry_after)
    
    # Exponential backoff with jitter
    delay = base_delay * (2 ** (attempt - 1))
    
    # Add jitter (±25%)
    import random
    jitter = random.uniform(0.75, 1.25)
    
    return min(delay * jitter, 60.0)  # Cap at 60 seconds 