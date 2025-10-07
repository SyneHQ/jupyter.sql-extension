"""
SQL Service Client Module

This module provides the client interface for communicating with SQL services,
handling connection management, authentication, and query execution.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import aiohttp
from aiohttp.web import head
from pydantic import BaseModel, Field, validator

from .exceptions import (
    ConnectionError,
    AuthenticationError,
    QueryExecutionError,
    ValidationError
)
from .types import QueryResult, QueryMetadata


logger = logging.getLogger(__name__)


@dataclass
class ConnectionConfig:
    """Configuration for database connections."""
    
    connection_id: str
    type: str = "postgresql"  # Database type
    host: str = ""
    port: str = ""  # Changed to string to match API
    database: str = ""
    username: str = ""
    password: str = ""
    schema: str = ""
    table: str = ""
    ssl_mode: str = "prefer"
    file_path: str = "/app/syne-dummy-db.duckdb"  # Default file path
    additional_params: Dict[str, str] = field(default_factory=dict)
    application_name: str = ""
    account: str = ""
    warehouse: str = ""
    role: str = ""
    project_id: str = ""
    key_file: str = ""
    region: str = ""
    atlas_srv: bool = False
    
    # Additional client-specific parameters
    connection_timeout: int = 30
    query_timeout: int = 300
    max_connections: int = 10
    min_connections: int = 1
    
    # Connection metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate connection configuration after initialization."""
        if not self.connection_id:
            raise ValidationError("Connection ID is required")
        if not self.type:
            raise ValidationError("Database type is required")
        
        # Type-specific validation
        if self.type in ["postgresql", "mysql", "mariadb", "sqlserver", "clickhouse", "oracle"]:
            if not self.host:
                raise ValidationError(f"Host is required for {self.type}")
            if not self.database:
                raise ValidationError(f"Database name is required for {self.type}")
            if not self.username:
                raise ValidationError(f"Username is required for {self.type}")
            if not self.password:
                raise ValidationError(f"Password is required for {self.type}")
        elif self.type in ["sqlite", "duckdb", "csv"]:
            if not self.file_path:
                raise ValidationError(f"File path is required for {self.type}")
    
    def get_connection_string(self) -> str:
        """Generate connection string for the database."""
        if self.type == "postgresql":
            port = self.port or "5432"
            return f"postgresql://{self.username}:{self.password}@{self.host}:{port}/{self.database}"
        elif self.type in ["mysql", "mariadb"]:
            port = self.port or "3306"
            return f"mysql://{self.username}:{self.password}@{self.host}:{port}/{self.database}"
        elif self.type == "sqlite":
            return f"sqlite:///{self.file_path}"
        elif self.type == "duckdb":
            return f"duckdb:///{self.file_path}"
        else:
            port = self.port or "5432"
            return f"{self.type}://{self.username}:{self.password}@{self.host}:{port}/{self.database}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation matching the API structure."""
        return {
            "type": self.type,
            "host": self.host,
            "port": str(self.port),
            "username": self.username,
            "password": self.password,
            "database": self.database,
            "schema": self.schema,
            "table": self.table,
            "ssl_mode": self.ssl_mode,
            "file_path": self.file_path,
            "additional_params": self.additional_params,
            "application_name": self.application_name,
            "account": self.account,
            "warehouse": self.warehouse,
            "role": self.role,
            "project_id": self.project_id,
            "key_file": self.key_file,
            "region": self.region,
            "atlas_srv": self.atlas_srv
        }
    
    def to_safe_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation without sensitive data."""
        safe_dict = self.to_dict()
        safe_dict["password"] = "***" if self.password else ""
        safe_dict["key_file"] = "***" if self.key_file else ""
        return safe_dict


class SQLServiceClient:
    """
    Client for communicating with SQL services.
    
    This client handles:
    - Connection configuration retrieval
    - Query execution
    - Authentication and authorization
    - Error handling and retries
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        session: Optional[aiohttp.ClientSession] = None
    ):
        """
        Initialize the SQL service client.
        
        Args:
            base_url: Base URL of the SQL service
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            session: Optional aiohttp session for connection pooling
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session
        self._headers = self._build_headers()
        
        logger.info("SQLServiceClient initialized for %s", base_url)
    
    def _build_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "JupyterSQLExtension/1.0.0"
        }
        
        if self.api_key:
            headers["X-Api-Key"] = self.api_key
        
        return headers
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            # Don't set session-level headers to avoid Content-Type conflicts
            # Headers will be set per-request instead
            # Use custom connector to handle duplicate headers gracefully
            connector = aiohttp.TCPConnector()
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                # Enable response compression
                auto_decompress=True
            )
        return self._session
    
    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        verbose: bool = False,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to the SQL service.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request data
            params: Query parameters
            
        Returns:
            Response data
            
        Raises:
            ConnectionError: If connection fails
            AuthenticationError: If authentication fails
            QueryExecutionError: If query execution fails
        """
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        # Merge default headers with request-specific headers
        request_headers = self._headers.copy()
        if headers:
            request_headers.update(headers)
        
        if verbose:
            print(f"Making request to {url} with method {method} and data {data}")
        
        for attempt in range(self.max_retries + 1):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=request_headers,
                ) as response:
                    # Debug logging for response headers
                    if verbose:
                        content_type = response.headers.get('content-type', 'unknown')
                        logger.debug("Response status: %s, Content-Type: %s", response.status, content_type)
                        logger.debug("Response headers: %s", dict(response.headers))
                        # Check for duplicate or conflicting Content-Type headers (common server issue)
                        content_type_headers = response.headers.getall('content-type', [])
                        if len(content_type_headers) > 1:
                            logger.warning("Server sent multiple Content-Type headers: %s", content_type_headers)
                            # Check if we have conflicting headers (e.g., text/plain + application/json)
                            has_json = any('application/json' in ct.lower() for ct in content_type_headers)
                            has_plain = any('text/plain' in ct.lower() for ct in content_type_headers)
                            if has_json and has_plain:
                                logger.warning("Server sent conflicting Content-Type headers - will attempt JSON parsing anyway")
                    
                    # Check for successful status codes
                    if response.status == 200:
                        # Check if we have conflicting headers that indicate JSON despite aiohttp's interpretation
                        content_type_headers = response.headers.getall('content-type', [])
                        should_force_json = False
                        
                        if len(content_type_headers) > 1:
                            has_json = any('application/json' in ct.lower() for ct in content_type_headers)
                            has_plain = any('text/plain' in ct.lower() for ct in content_type_headers)
                            if has_json and has_plain:
                                should_force_json = True
                                logger.debug("Detected conflicting Content-Type headers, forcing JSON parsing")
                        
                        # Try aiohttp's built-in JSON parsing first
                        if not should_force_json:
                            try:
                                return await response.json()
                            except aiohttp.ContentTypeError:
                                should_force_json = True
                                logger.debug("aiohttp Content-Type validation failed, falling back to manual JSON parsing")
                        
                        # Force JSON parsing (either due to conflicting headers or aiohttp failure)
                        if should_force_json:
                            response_text = await response.text()
                            try:
                                parsed_json = json.loads(response_text)
                                logger.debug("Successfully parsed JSON response manually")
                                return parsed_json
                            except json.JSONDecodeError as json_err:
                                content_type = response.headers.get('content-type', 'unknown')
                                logger.error("Failed to parse JSON response. Content-Type: %s, Error: %s", content_type, json_err)
                                logger.error("Response text (first 500 chars): %s", response_text[:500])
                                raise QueryExecutionError(f"Server returned invalid JSON response. Content-Type: {content_type}")
                    elif response.status == 401:
                        logging.debug(f"Authentication error: {response.status}")
                        raise AuthenticationError("Invalid API key or authentication failed")
                    elif response.status == 403:
                        logging.debug(f"Insufficient permissions: {response.status}")
                        raise AuthenticationError("Insufficient permissions")
                    elif response.status == 404:
                        logging.debug(f"Endpoint not found: {endpoint}")
                        raise ConnectionError(f"Endpoint not found: {endpoint}")
                    elif response.status == 500:
                        try:
                            error_data = await response.json()
                            print("🚀 ~ _make_request ~ error_data:", error_data)
                            if 'details' in error_data:
                                error_message = error_data.get('details', 'Unknown error')
                            else:
                                error_message = error_data.get('error', 'Unknown error')
                        except (aiohttp.ContentTypeError, json.JSONDecodeError):
                            error_message = await response.text()
                            print("🚀 ~ _make_request ~ error_message:", error_message)
                        raise QueryExecutionError(f"Server error: {error_message}")
                    else:
                        try:
                            error_data = await response.json()
                            if 'details' in error_data:
                                error_message = error_data.get('details', 'Unknown error')
                            else:
                                error_message = error_data.get('error', 'Unknown error')
                        except (aiohttp.ContentTypeError, json.JSONDecodeError):
                            error_message = await response.text()
                        raise ConnectionError(f"HTTP {response.status}: {error_message}")
                        
            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    raise ConnectionError(f"Failed to connect to SQL service after {self.max_retries} attempts: {e}")
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except asyncio.TimeoutError:
                if attempt == self.max_retries:
                    raise ConnectionError(f"Request timeout after {self.max_retries} attempts")
                logger.warning(f"Request attempt {attempt + 1} timed out")
                await asyncio.sleep(2 ** attempt)
    
    async def execute_query(
        self,
        connection_id: str,
        query: str,
        verbose: bool = False,
        timeout: Optional[int] = None,
        explain: bool = False,
        api_key: str = None,
        **kwargs: Any
    ) -> QueryResult:
        """
        Execute SQL query against the database.
        
        Args:
            connection_config: Database connection configuration
            query: SQL query to execute
            timeout: Query timeout in seconds
            explain: Whether to return execution plan
            api_key: API key for the SQL service
        Returns:
            Query execution result
            
        Raises:
            QueryExecutionError: If query execution fails
            ConnectionError: If connection fails
        """
        try:
            # Prepare request data
            request_data = {
                "id": connection_id,
                "query": query,
                "transaction": True
            }

            request_data.update(kwargs)
            
            # Execute query
            headers = {}
            if api_key:
                headers["X-Api-Key"] = api_key
            
            response = await self._make_request(
                method="POST",
                endpoint="/magic.query",
                data=request_data,
                headers=headers,
                verbose=verbose
            )
            
            # Parse response - API returns QueryResponse format
            if "error" in response and response["error"]:
                raise QueryExecutionError(response["error"])
            
            # Extract results from QueryResponse format
            results = response.get("results", [])
            elapsed = response.get("elapsed", 0.0)
            row_count = response.get("rowCount", 0)
            
            # Convert results to rows and columns format
            if results and row_count > 0:
                # Extract column names from first result object
                columns = list(results[0].keys()) if results else []
                # Convert to row format
                data = [[row.get(col, None) for col in columns] for row in results]
            else:
                columns = []
                data = []
            
            # Create QueryResult
            result = QueryResult(
                data=data,
                columns=columns,
                row_count=row_count,
                metadata=QueryMetadata(
                    connection_id=connection_id,
                    query_hash=hash(query),
                    execution_time=elapsed,
                    row_count=row_count,
                    success=True,
                    timestamp=datetime.now().timestamp()
                )
            )
            
            return result
            
        except Exception as e:
            if isinstance(e, (QueryExecutionError, ConnectionError)):
                raise
            raise QueryExecutionError(f"Failed to execute query: {e}")
    
    async def list_connections(self, api_key) -> List[str]:
        """
        List all connections.
        """
        response = await self._make_request(
            method="GET",
            endpoint="/connections",
            headers={
                'X-Api-Key': api_key
            }
        )
        return response.get("connections", [])
    
    async def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate SQL query without execution.
        
        Args:
            query: SQL query to validate
            
        Returns:
            Validation result
        """
        try:
            response = await self._make_request(
                method="POST",
                endpoint="/api/query/validate",
                data={"query": query}
            )
            
            return response.get("data", {})
            
        except Exception as e:
            raise QueryExecutionError(f"Failed to validate query: {e}")

# Convenience function for creating client
def create_sql_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
    max_retries: int = 3
) -> SQLServiceClient:
    """
    Create a new SQL service client.
    
    Args:
        base_url: Base URL of the SQL service
        api_key: API key for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Configured SQL service client
    """
    return SQLServiceClient(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries
    ) 