"""
Jupyter SQL Extension - Magic Command Implementation

This module implements the %%sqlconnect cell magic for secure SQL query execution
through internal service connections.

The magic provides:
- Secure connection management with credential retrieval
- Input validation and SQL injection prevention
- Rich output formatting (DataFrames, tables, JSON)
- Comprehensive error handling with user-friendly messages
- Type-safe implementation with full logging
- Modular, testable architecture
"""

import asyncio
import logging
import re
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from functools import wraps
from urllib.parse import urlparse

# IPython/Jupyter imports
from IPython.core.magic import Magics, cell_magic, magics_class, needs_local_scope
from IPython.core.magic_arguments import (
    argument,
    magic_arguments,
    parse_argstring,
    MagicArgumentParser
)
from IPython.display import display, HTML, JSON
from IPython.core.display import DisplayObject

# Data handling
import pandas as pd

# Our extension modules
from .client import SQLServiceClient, ConnectionConfig
from .config import ExtensionConfig, load_config
from .exceptions import (
    SQLExtensionError,
    ConnectionError,
    AuthenticationError,
    ValidationError,
    QueryExecutionError,
    ConfigurationError
)
from .utils import (
    sanitize_query,
    validate_connection_id,
    format_query_result,
    setup_logging,
    measure_performance
)
from .types import QueryResult, OutputFormat, QueryMetadata


# Set up module logger
logger = logging.getLogger(__name__)





def async_cell_magic(func):
    """
    Decorator to enable async/await in cell magics.

    Jupyter cell magics don't natively support async, so this decorator
    runs async functions in the event loop, handling both running and non-running loops.
    """
    @wraps(func)
    def wrapper(self, line: str, cell: str) -> Any:
        try:
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in a running loop - need to handle this carefully
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(func(self, line, cell))
                except ImportError:
                    # nest_asyncio not available, use thread executor
                    import concurrent.futures
                    
                    def run_async():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(func(self, line, cell))
                        finally:
                            try:
                                # Clean up any pending tasks
                                pending = asyncio.all_tasks(new_loop)
                                for task in pending:
                                    task.cancel()
                                new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            except Exception:
                                pass
                            finally:
                                new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_async)
                        return future.result()
            except RuntimeError:
                # No running loop - safe to use asyncio.run
                return asyncio.run(func(self, line, cell))
        except Exception as e:
            logger.error(f"Error in async magic execution: {e}")
            raise

    return wrapper


@magics_class
class SQLConnectMagic(Magics):
    # Override the config trait to avoid traitlets conflicts
    config = None
    """
    Jupyter magic for executing SQL queries through internal services.

    This magic provides the %%sqlconnect command that:
    1. Takes a connectionId to retrieve database credentials securely
    2. Executes SQL queries against the configured database
    3. Returns rich formatted results (DataFrames, HTML tables, JSON)
    4. Handles errors gracefully with detailed feedback

    Example usage:
        %%sqlconnect my_db_connection
        SELECT * FROM users
        WHERE created_at >= '2024-01-01'
        LIMIT 10
    """

    def __init__(self, shell=None):
        """
        Initialize the SQL magic with configuration and clients.

        Args:
            shell: IPython shell instance
        """
        # Initialize _config first to avoid property access during initialization
        try:
            self._config = load_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config = ExtensionConfig()  # Use defaults

        # Call parent constructor after config is set
        super().__init__(shell)

        # Initialize SQL service client
        self.sql_client = SQLServiceClient(
            base_url=self._config.sql_service_url,
            api_key=self._config.sql_service_api_key,
            timeout=self._config.default_timeout,
            max_retries=self._config.max_retries
        )

        # Setup logging
        if self._config.enable_logging:
            setup_logging(
                level=self._config.log_level,
                log_file=self._config.log_file
            )

        # Cache for connection configurations
        self._connection_cache: Dict[str, ConnectionConfig] = {}

        # Performance tracking
        self._query_metrics: List[QueryMetadata] = []

        logger.info("SQLConnectMagic initialized successfully")

        # print the config
        logger.info(f"SQLConnectMagic initialized with config: {self._config}")
    
    def _load_config(self, cfg):
        """Override to avoid traitlets config loading issues."""
        # Skip traitlets config loading since we use our own config system
        pass
    
    def __del__(self):
        """Cleanup when the magic is destroyed."""
        try:
            if hasattr(self, 'sql_client'):
                # Use thread executor for cleanup to avoid event loop conflicts
                import concurrent.futures
                
                def cleanup():
                    try:
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self.sql_client.close())
                        finally:
                            try:
                                # Clean up any pending tasks
                                pending = asyncio.all_tasks(new_loop)
                                for task in pending:
                                    task.cancel()
                                new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            except Exception:
                                pass
                            finally:
                                new_loop.close()
                    except Exception:
                        pass  # Ignore cleanup errors
                
                # Run cleanup in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    executor.submit(cleanup)
        except Exception:
            pass  # Ignore cleanup errors



    @cell_magic
    @magic_arguments()
    @argument(
        'connection_id',
        type=str,
        help='Database connection identifier, "direct" for default connection, or "config:..." for custom config'
    )

    @argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of rows to return'
    )
    @argument(
        '--timeout', '-t',
        type=int,
        default=None,
        help='Query timeout in seconds'
    )
    @argument(
        '--no-cache',
        action='store_true',
        help='Disable connection caching'
    )
    @argument(
        '--explain',
        action='store_true',
        help='Show query execution plan instead of results'
    )
    @argument(
        '--dry-run',
        action='store_true',
        help='Validate query without execution'
    )
    @argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    def sqlconnect(self, line: str, cell: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query through secure connection service.

        This magic command:
        1. Parses the connection ID and arguments
        2. Validates and sanitizes the SQL query
        3. Retrieves connection configuration securely
        4. Executes the query against the database
        5. Returns a pandas DataFrame with the results

        Args:
            line: Magic command line with arguments
            cell: SQL query content

        Returns:
            Pandas DataFrame with query results or None on error
        """
        # Run the async implementation safely
        try:
            # Try to use nest_asyncio first if available
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self._sqlconnect_async(line, cell))
            except ImportError:
                # nest_asyncio not available, check if we're in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # We're in a running loop - use thread executor to avoid conflicts
                    import concurrent.futures
                    
                    def run_in_thread():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self._sqlconnect_async(line, cell))
                        finally:
                            try:
                                # Clean up any pending tasks
                                pending = asyncio.all_tasks(new_loop)
                                for task in pending:
                                    task.cancel()
                                new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                            except Exception:
                                pass
                            finally:
                                new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        return future.result()
                except RuntimeError:
                    # No running loop - safe to use asyncio.run
                    return asyncio.run(self._sqlconnect_async(line, cell))
        except Exception as e:
            logger.error(f"Error in sqlconnect magic: {e}")
            self._display_error("Unexpected Error", str(e), "error")
            return None

    async def _sqlconnect_async(self, line: str, cell: str) -> Optional[pd.DataFrame]:
        """
        Async implementation of the sqlconnect magic.
        
        Args:
            line: Magic command line with arguments
            cell: SQL query content

        Returns:
            Pandas DataFrame with query results or None on error
        """
        start_time = time.time()
        connection_id = None

        try:
            # Parse magic arguments
            args = parse_argstring(self.sqlconnect, line)

            if args.verbose:
                logger.setLevel(logging.DEBUG)
                print("🔍 Verbose mode enabled")

            # Validate connection ID
            connection_id = args.connection_id.strip()
            if not validate_connection_id(connection_id):
                raise ValidationError(f"Invalid connection ID: {connection_id}")

            # Get local namespace for variable substitution
            local_ns = self.shell.user_ns if self.shell else {}

            # Validate and sanitize SQL query
            sql_query = self._prepare_query(cell, local_ns, args.verbose)

            # Dry run mode - just validate the query
            if args.dry_run:
                self._display_dry_run_result(connection_id, sql_query)
                return None
            
            # Get connection configuration
            connection_config = await self._get_connection_config(
                connection_id,
                use_cache=not args.no_cache,
                verbose=args.verbose
            )

            # Execute the query
            result = await self._execute_query(
                connection_config=connection_config,
                query=sql_query,
                limit=args.limit,
                timeout=args.timeout or self._config.query_timeout,
                explain=args.explain,
                verbose=args.verbose
            )

            # Record performance metrics
            execution_time = time.time() - start_time
            self._record_metrics(
                connection_id=connection_id,
                query=sql_query,
                execution_time=execution_time,
                row_count=len(result.data) if result.data else 0,
                success=True
            )

            if args.verbose:
                print(f"✅ Query executed successfully in {execution_time:.2f}s")

            # Return DataFrame
            if not result.data:
                if args.verbose:
                    print("📊 Query returned no results")
                return pd.DataFrame()  # Empty DataFrame
            
            # Create DataFrame from result
            df = pd.DataFrame(result.data, columns=result.columns)
            
            # Display metadata if verbose
            if args.verbose:
                print(f"📊 Query returned {len(df)} rows with {len(df.columns)} columns")
                if result.metadata:
                    execution_time = result.metadata.get('execution_time', 'unknown')
                    print(f"📊 Execution time: {execution_time}")
            
            return df

        except ValidationError as e:
            logger.warning(f"Validation error: {e}")
            self._display_error("Validation Error", str(e), "warning")
            self._record_metrics(connection_id or "unknown", cell, time.time() - start_time, 0, False)

        except AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            self._display_error("Authentication Error",
                              "Failed to authenticate with the SQL service. Please check your credentials.",
                              "error")
            self._record_metrics(connection_id or "unknown", cell, time.time() - start_time, 0, False)

        except ConnectionError as e:
            logger.error(f"Connection error: {e}")
            self._display_error("Connection Error",
                              f"Unable to connect to database '{connection_id}'. Please verify the connection exists and is accessible.",
                              "error")
            self._record_metrics(connection_id or "unknown", cell, time.time() - start_time, 0, False)

        except QueryExecutionError as e:
            logger.error(f"Query execution error: {e}")
            self._display_error("Query Execution Error", str(e), "error")
            self._record_metrics(connection_id or "unknown", cell, time.time() - start_time, 0, False)

        except Exception as e:
            logger.exception(f"Unexpected error in sqlconnect magic: {e}")
            self._display_error("Unexpected Error",
                              f"An unexpected error occurred: {str(e)}",
                              "error")
            self._record_metrics(connection_id or "unknown", cell, time.time() - start_time, 0, False)

        return None

    def _prepare_query(self, cell: str, local_ns: Optional[Dict[str, Any]], verbose: bool = False) -> str:
        """
        Prepare and validate the SQL query.

        Args:
            cell: Raw SQL query from cell
            local_ns: Local namespace for variable substitution
            verbose: Enable verbose output

        Returns:
            Sanitized and validated SQL query

        Raises:
            ValidationError: If query validation fails
        """
        if not cell or not cell.strip():
            raise ValidationError("SQL query cannot be empty")

        # Basic SQL injection prevention
        query = sanitize_query(cell)

        if verbose:
            print(f"📝 Original query length: {len(cell)} characters")
            print(f"📝 Sanitized query length: {len(query)} characters")

        # Variable substitution from local namespace
        if local_ns:
            query = self._substitute_variables(query, local_ns, verbose)

        # Additional validation
        self._validate_query_safety(query)

        return query

    def _substitute_variables(self, query: str, local_ns: Dict[str, Any], verbose: bool = False) -> str:
        """
        Substitute Python variables in the SQL query safely.

        Args:
            query: SQL query with potential variable references
            local_ns: Local namespace containing variables
            verbose: Enable verbose output

        Returns:
            Query with variables substituted
        """
        # Find variable references like {variable_name}
        variable_pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        variables_found = re.findall(variable_pattern, query)

        if variables_found and verbose:
            print(f"🔄 Found variables to substitute: {variables_found}")

        for var_name in variables_found:
            if var_name in local_ns:
                var_value = local_ns[var_name]

                # Safe string formatting for different types
                if isinstance(var_value, str):
                    # Escape single quotes and wrap in quotes
                    safe_value = f"'{var_value.replace(chr(39), chr(39) + chr(39))}'"
                elif isinstance(var_value, (int, float)):
                    safe_value = str(var_value)
                elif isinstance(var_value, (list, tuple)):
                    # Convert to SQL IN clause format
                    safe_items = [f"'{str(item).replace(chr(39), chr(39) + chr(39))}'" if isinstance(item, str) else str(item) for item in var_value]
                    safe_value = f"({', '.join(safe_items)})"
                else:
                    safe_value = f"'{str(var_value).replace(chr(39), chr(39) + chr(39))}'"

                query = query.replace(f"{{{var_name}}}", safe_value)

                if verbose:
                    print(f"🔄 Substituted {var_name} = {safe_value}")
            else:
                raise ValidationError(f"Variable '{var_name}' not found in local namespace")

        return query

    def _validate_query_safety(self, query: str) -> None:
        """
        Validate query for basic safety checks.

        Args:
            query: SQL query to validate

        Raises:
            ValidationError: If query fails safety checks
        """
        query_upper = query.upper().strip()

        # Check for dangerous operations
        dangerous_keywords = [
            'DROP ', 'DELETE ', 'TRUNCATE ', 'ALTER ', 'CREATE ',
            'INSERT ', 'UPDATE ', 'GRANT ', 'REVOKE ', 'EXEC',
            'EXECUTE ', 'xp_', 'sp_'
        ]

        for keyword in dangerous_keywords:
            if keyword in query_upper:
                if not self._config.allow_write_operations:
                    raise ValidationError(f"Write operations are not allowed: {keyword.strip()}")

        # Check query length
        if len(query) > self._config.max_query_length:
            raise ValidationError(f"Query too long: {len(query)} characters (max: {self._config.max_query_length})")

        # Basic SQL syntax validation
        if not re.search(r'\bSELECT\b', query_upper):
            if not self._config.allow_non_select_queries:
                raise ValidationError("Only SELECT queries are allowed")

    async def _get_connection_config(self, connection_id: str, use_cache: bool = True, verbose: bool = False) -> ConnectionConfig:
        """
        Retrieve connection configuration securely.

        Args:
            connection_id: Database connection identifier
            use_cache: Whether to use cached configurations
            verbose: Enable verbose output

        Returns:
            Connection configuration

        Raises:
            ConnectionError: If connection config retrieval fails
        """
        if use_cache and connection_id in self._connection_cache:
            if verbose:
                print(f"📋 Using cached connection config for '{connection_id}'")
            return self._connection_cache[connection_id]

        if verbose:
            print(f"🔗 Retrieving connection config for '{connection_id}'")

        try:
            connection_config = await self.sql_client.get_connection_config(connection_id)

            if use_cache:
                self._connection_cache[connection_id] = connection_config
                if verbose:
                    print(f"📋 Cached connection config for '{connection_id}'")

            return connection_config

        except Exception as e:
            raise ConnectionError(f"Failed to retrieve connection configuration: {e}")

    @measure_performance
    async def _execute_query(
        self,
        connection_config: ConnectionConfig,
        query: str,
        limit: Optional[int] = None,
        timeout: int = 30,
        explain: bool = False,
        verbose: bool = False
    ) -> QueryResult:
        """
        Execute SQL query against the database.

        Args:
            connection_config: Database connection configuration
            query: SQL query to execute
            limit: Maximum number of rows to return
            timeout: Query timeout in seconds
            explain: Whether to return execution plan
            verbose: Enable verbose output

        Returns:
            Query execution result

        Raises:
            QueryExecutionError: If query execution fails
        """
        if verbose:
            print(f"⚡ Executing query (timeout: {timeout}s, limit: {limit or 'none'})")

        try:
            # Apply limit if specified
            if limit and not explain:
                query = self._apply_query_limit(query, limit)

            # Execute the query
            result = await self.sql_client.execute_query(
                connection_config=connection_config,
                query=query,
                timeout=timeout,
                explain=explain
            )

            if verbose:
                rows = len(result.data) if result.data else 0
                print(f"📊 Query returned {rows} rows")
                if result.metadata:
                    print(f"📊 Execution time: {result.metadata.get('execution_time', 'unknown')}")

            return result

        except Exception as e:
            raise QueryExecutionError(f"Query execution failed: {e}")

    def _apply_query_limit(self, query: str, limit: int) -> str:
        """
        Apply LIMIT clause to query if not already present.

        Args:
            query: Original SQL query
            limit: Maximum number of rows

        Returns:
            Query with LIMIT clause
        """
        query_upper = query.upper()

        # Check if LIMIT already exists
        if 'LIMIT' in query_upper:
            return query  # Don't modify if LIMIT already present

        # Add LIMIT clause
        return f"{query.rstrip(';')} LIMIT {limit}"

    def _format_result(self, result: QueryResult, output_format: str, verbose: bool = False) -> DisplayObject:
        """
        Format query result for display.

        Args:
            result: Query execution result
            output_format: Desired output format
            verbose: Enable verbose output

        Returns:
            Formatted display object
        """
        if not result.data:
            return HTML("<p><em>Query returned no results.</em></p>")

        if verbose:
            print(f"🎨 Formatting {len(result.data)} rows as {output_format}")

        try:
            if output_format == 'dataframe':
                return self._format_as_dataframe(result)
            elif output_format == 'json':
                return self._format_as_json(result)
            elif output_format == 'html':
                return self._format_as_html(result)
            else:  # table (default)
                return self._format_as_table(result)
        except Exception as e:
            logger.warning(f"Failed to format result as {output_format}: {e}")
            # Fallback to simple table
            return self._format_as_table(result)

    def _format_as_dataframe(self, result: QueryResult) -> pd.DataFrame:
        """Format result as pandas DataFrame."""
        return pd.DataFrame(result.data, columns=result.columns)

    def _format_as_json(self, result: QueryResult) -> JSON:
        """Format result as JSON display object."""
        data = {
            'columns': result.columns,
            'data': result.data,
            'row_count': len(result.data),
            'metadata': result.metadata
        }
        return JSON(data)

    def _format_as_html(self, result: QueryResult) -> HTML:
        """Format result as HTML table."""
        df = pd.DataFrame(result.data, columns=result.columns)
        html_table = df.to_html(
            classes='table table-striped table-hover',
            table_id='sql-result-table',
            escape=False,
            max_rows=self._config.max_display_rows
        )

        # Add metadata
        metadata_html = ""
        if result.metadata:
            execution_time = result.metadata.get('execution_time', 'unknown')
            metadata_html = f"""
            <div class="sql-metadata" style="margin-top: 10px; color: #666; font-size: 0.9em;">
                <strong>Rows:</strong> {len(result.data)} |
                <strong>Execution Time:</strong> {execution_time}
            </div>
            """

        return HTML(f"""
        <div class="sql-result">
            {html_table}
            {metadata_html}
        </div>
        """)

    def _format_as_table(self, result: QueryResult) -> HTML:
        """Format result as simple HTML table."""
        if not result.data:
            return HTML("<p><em>No results</em></p>")

        # Build table HTML
        html_parts = ['<table class="table table-striped">']

        # Header
        html_parts.append('<thead><tr>')
        for col in result.columns:
            html_parts.append(f'<th>{col}</th>')
        html_parts.append('</tr></thead>')

        # Body
        html_parts.append('<tbody>')
        max_rows = min(len(result.data), self._config.max_display_rows)
        for i in range(max_rows):
            row = result.data[i]
            html_parts.append('<tr>')
            for value in row:
                # Handle None values and convert to string safely
                display_value = str(value) if value is not None else ''
                html_parts.append(f'<td>{display_value}</td>')
            html_parts.append('</tr>')
        html_parts.append('</tbody>')
        html_parts.append('</table>')

        # Add truncation notice if needed
        if len(result.data) > self._config.max_display_rows:
            html_parts.append(f'<p><em>Showing {max_rows} of {len(result.data)} rows</em></p>')

        return HTML(''.join(html_parts))

    def _display_dry_run_result(self, connection_id: str, query: str) -> HTML:
        """Display dry run validation result."""
        return HTML(f"""
        <div class="alert alert-info">
            <h4>🔍 Dry Run - Query Validation</h4>
            <p><strong>Connection ID:</strong> {connection_id}</p>
            <p><strong>Query Length:</strong> {len(query)} characters</p>
            <p><strong>Status:</strong> ✅ Query passed validation</p>
            <details>
                <summary>View Query</summary>
                <pre><code>{query}</code></pre>
            </details>
        </div>
        """)

    def _display_error(self, error_type: str, message: str, level: str = "error") -> None:
        """
        Display error message with appropriate styling.

        Args:
            error_type: Type of error
            message: Error message
            level: Error level (error, warning, info)
        """
        icons = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
        colors = {"error": "#d32f2f", "warning": "#f57c00", "info": "#1976d2"}

        icon = icons.get(level, "❌")
        color = colors.get(level, "#d32f2f")

        error_html = f"""
        <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0; background-color: #f5f5f5;">
            <strong style="color: {color};">{icon} {error_type}</strong><br>
            {message}
        </div>
        """

        display(HTML(error_html))

    def _record_metrics(self, connection_id: str, query: str, execution_time: float, row_count: int, success: bool) -> None:
        """
        Record query execution metrics.

        Args:
            connection_id: Database connection identifier
            query: SQL query executed
            execution_time: Time taken to execute
            row_count: Number of rows returned
            success: Whether execution was successful
        """
        metrics = QueryMetadata(
            connection_id=connection_id,
            query_hash=hash(query),
            execution_time=execution_time,
            row_count=row_count,
            success=success,
            timestamp=time.time()
        )

        self._query_metrics.append(metrics)

        # Keep only recent metrics (last 100)
        if len(self._query_metrics) > 100:
            self._query_metrics = self._query_metrics[-100:]

    def get_metrics(self) -> List[QueryMetadata]:
        """Get query execution metrics."""
        return self._query_metrics.copy()

    def clear_cache(self) -> None:
        """Clear connection configuration cache."""
        self._connection_cache.clear()
        logger.info("Connection cache cleared")

    def get_cached_connections(self) -> List[str]:
        """Get list of cached connection IDs."""
        return list(self._connection_cache.keys())

def register_magic() -> bool:
    """
    Register the SQLConnect magic with IPython.

    Returns:
        True if registration successful, False otherwise
    """
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            logger.warning("Not in IPython environment, cannot register magic")
            return False

        # Try different registration methods based on IPython environment
        try:
            # Method 1: Use register_magic_class (available in newer IPython/Jupyter)
            if hasattr(ipython, 'register_magic_class'):
                ipython.register_magic_class(SQLConnectMagic)
                logger.info("SQLConnect magic registered using register_magic_class")
                return True
        except Exception as e:
            logger.debug(f"register_magic_class failed: {e}")

        try:
            # Method 2: Use register_magic_function (fallback for older IPython)
            if hasattr(ipython, 'register_magic_function'):
                # Create an instance and register the method
                magic_instance = SQLConnectMagic(ipython)
                ipython.register_magic_function(
                    magic_instance.sqlconnect,
                    magic_kind='cell',
                    magic_name='sqlconnect'
                )
                logger.info("SQLConnect magic registered using register_magic_function")
                return True
        except Exception as e:
            logger.debug(f"register_magic_function failed: {e}")

        try:
            # Method 3: Use magics_manager directly (most compatible)
            if hasattr(ipython, 'magics_manager'):
                magic_instance = SQLConnectMagic(ipython)
                ipython.magics_manager.magics['cell']['sqlconnect'] = magic_instance.sqlconnect
                logger.info("SQLConnect magic registered using magics_manager")
                return True
        except Exception as e:
            logger.debug(f"magics_manager registration failed: {e}")

        # If all methods fail, log the error
        logger.error("All magic registration methods failed")
        return False

    except Exception as e:
        logger.error(f"Failed to register magic: {e}")
        return False


# Auto-register when module is imported in IPython
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        register_magic()
except ImportError:
    pass  # Not in IPython environment
