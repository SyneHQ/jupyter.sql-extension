"""
Configuration Management for Jupyter SQL Extension

This module handles loading, validating, and managing configuration
from various sources including environment variables, config files, and defaults.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from functools import lru_cache

from .exceptions import ConfigurationError
from .types import ExtensionConfig, OutputFormat


logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration loading and validation.
    
    This class handles loading configuration from multiple sources:
    - Environment variables
    - Configuration files (JSON, YAML)
    - Default values
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self._config_cache: Optional[ExtensionConfig] = None
    
    def load_config(self) -> ExtensionConfig:
        """
        Load configuration from all available sources.
        
        Returns:
            Loaded configuration
            
        Raises:
            ConfigurationError: If configuration loading fails
        """
        try:
            # Start with defaults
            config = self._get_default_config()
            
            # Load from file if available
            if self.config_path:
                file_config = self._load_from_file(self.config_path)
                config = self._merge_configs(config, file_config)
            
            # Load from environment variables
            env_config = self._load_from_environment()
            config = self._merge_configs(config, env_config)
            
            # Validate configuration
            self._validate_config(config)
            
            self._config_cache = config
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def _get_default_config(self) -> ExtensionConfig:
        """Get default configuration."""
        return ExtensionConfig()
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If file loading fails
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Configuration file not found: {file_path}")
                return {}
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ConfigurationError(f"Unsupported configuration file format: {path.suffix}")
                    
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration file {file_path}: {e}")
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary
        """
        config = {
            'sql_service_url': 'https://cosmos.synehq.com/api/v1'
        }
        
        # Service configuration
        if os.getenv('SQL_SERVICE_URL'):
            config['sql_service_url'] = os.getenv('SQL_SERVICE_URL')
        if os.getenv('SQL_SERVICE_API_KEY'):
            config['sql_service_api_key'] = os.getenv('SQL_SERVICE_API_KEY')
        
        # Connection settings
        if os.getenv('DEFAULT_TIMEOUT'):
            config['default_timeout'] = int(os.getenv('DEFAULT_TIMEOUT'))
        if os.getenv('MAX_RETRIES'):
            config['max_retries'] = int(os.getenv('MAX_RETRIES'))
        if os.getenv('CONNECTION_POOL_SIZE'):
            config['connection_pool_size'] = int(os.getenv('CONNECTION_POOL_SIZE'))
        
        # Query settings
        if os.getenv('QUERY_TIMEOUT'):
            config['query_timeout'] = int(os.getenv('QUERY_TIMEOUT'))
        if os.getenv('MAX_QUERY_LENGTH'):
            config['max_query_length'] = int(os.getenv('MAX_QUERY_LENGTH'))
        if os.getenv('ALLOW_WRITE_OPERATIONS'):
            config['allow_write_operations'] = os.getenv('ALLOW_WRITE_OPERATIONS').lower() == 'true'
        if os.getenv('ALLOW_NON_SELECT_QUERIES'):
            config['allow_non_select_queries'] = os.getenv('ALLOW_NON_SELECT_QUERIES').lower() == 'true'
        
        # Output settings
        if os.getenv('DEFAULT_OUTPUT_FORMAT'):
            config['default_output_format'] = os.getenv('DEFAULT_OUTPUT_FORMAT')
        if os.getenv('MAX_DISPLAY_ROWS'):
            config['max_display_rows'] = int(os.getenv('MAX_DISPLAY_ROWS'))
        if os.getenv('ENABLE_RICH_OUTPUT'):
            config['enable_rich_output'] = os.getenv('ENABLE_RICH_OUTPUT').lower() == 'true'
        
        # Security settings
        if os.getenv('ENABLE_SQL_VALIDATION'):
            config['enable_sql_validation'] = os.getenv('ENABLE_SQL_VALIDATION').lower() == 'true'
        if os.getenv('ENABLE_INPUT_SANITIZATION'):
            config['enable_input_sanitization'] = os.getenv('ENABLE_INPUT_SANITIZATION').lower() == 'true'
        
        # Logging settings
        if os.getenv('ENABLE_LOGGING'):
            config['enable_logging'] = os.getenv('ENABLE_LOGGING').lower() == 'true'
        if os.getenv('LOG_LEVEL'):
            config['log_level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            config['log_file'] = os.getenv('LOG_FILE')
        
        # Caching settings
        if os.getenv('ENABLE_CACHING'):
            config['enable_caching'] = os.getenv('ENABLE_CACHING').lower() == 'true'
        if os.getenv('CACHE_TTL'):
            config['cache_ttl'] = int(os.getenv('CACHE_TTL'))
        if os.getenv('MAX_CACHE_SIZE'):
            config['max_cache_size'] = int(os.getenv('MAX_CACHE_SIZE'))
        
        # Performance settings
        if os.getenv('ENABLE_METRICS'):
            config['enable_metrics'] = os.getenv('ENABLE_METRICS').lower() == 'true'
        if os.getenv('ENABLE_PROFILING'):
            config['enable_profiling'] = os.getenv('ENABLE_PROFILING').lower() == 'true'
        if os.getenv('MAX_CONCURRENT_QUERIES'):
            config['max_concurrent_queries'] = int(os.getenv('MAX_CONCURRENT_QUERIES'))
        
        return config
    
    def _merge_configs(self, base_config: ExtensionConfig, override_config: Dict[str, Any]) -> ExtensionConfig:
        """
        Merge configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to merge on top
            
        Returns:
            Merged configuration
        """
        # Convert base config to dict
        base_dict = base_config.to_dict()
        
        # Merge configurations
        merged_dict = {**base_dict, **override_config}
        
        # Convert back to ExtensionConfig
        return self._dict_to_config(merged_dict)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExtensionConfig:
        """
        Convert dictionary to ExtensionConfig.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExtensionConfig instance
        """
        # Handle special cases
        if 'default_output_format' in config_dict:
            try:
                config_dict['default_output_format'] = OutputFormat(config_dict['default_output_format'])
            except ValueError:
                logger.warning(f"Invalid output format: {config_dict['default_output_format']}, using default")
                config_dict['default_output_format'] = OutputFormat.TABLE
        
        # Create ExtensionConfig
        return ExtensionConfig(**config_dict)
    
    def _validate_config(self, config: ExtensionConfig) -> None:
        """
        Validate configuration values.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Validate URLs
        if not config.sql_service_url:
            raise ConfigurationError("SQL service URL is required")
        
        # Validate timeouts
        if config.default_timeout <= 0:
            raise ConfigurationError("Default timeout must be positive")
        if config.query_timeout <= 0:
            raise ConfigurationError("Query timeout must be positive")
        
        # Validate limits
        if config.max_retries < 0:
            raise ConfigurationError("Max retries must be non-negative")
        if config.connection_pool_size <= 0:
            raise ConfigurationError("Connection pool size must be positive")
        if config.max_query_length <= 0:
            raise ConfigurationError("Max query length must be positive")
        if config.max_display_rows <= 0:
            raise ConfigurationError("Max display rows must be positive")
        if config.cache_ttl <= 0:
            raise ConfigurationError("Cache TTL must be positive")
        if config.max_cache_size <= 0:
            raise ConfigurationError("Max cache size must be positive")
        if config.max_concurrent_queries <= 0:
            raise ConfigurationError("Max concurrent queries must be positive")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.log_level.upper() not in valid_log_levels:
            raise ConfigurationError(f"Invalid log level: {config.log_level}")
    
    def reload_config(self) -> ExtensionConfig:
        """
        Reload configuration from sources.
        
        Returns:
            Reloaded configuration
        """
        self._config_cache = None
        return self.load_config()
    
    def get_config(self) -> ExtensionConfig:
        """
        Get current configuration (cached if available).
        
        Returns:
            Current configuration
        """
        if self._config_cache is None:
            return self.load_config()
        return self._config_cache


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        Configuration manager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> ExtensionConfig:
    """
    Load configuration from all available sources.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Loaded configuration
    """
    manager = ConfigManager(config_path)
    return manager.load_config()


def validate_config(config: ExtensionConfig) -> bool:
    """
    Validate configuration values.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ConfigurationError: If validation fails
    """
    manager = ConfigManager()
    manager._validate_config(config)
    return True


def get_config_value(key: str, default: Any = None) -> Any:
    """
    Get a specific configuration value.
    
    Args:
        key: Configuration key
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    try:
        config = load_config()
        return getattr(config, key, default)
    except Exception as e:
        logger.warning(f"Failed to get config value for {key}: {e}")
        return default


def set_config_value(key: str, value: Any) -> None:
    """
    Set a configuration value (temporary, not persisted).
    
    Args:
        key: Configuration key
        value: Configuration value
    """
    try:
        config = load_config()
        if hasattr(config, key):
            setattr(config, key, value)
            logger.info(f"Set config value: {key} = {value}")
        else:
            logger.warning(f"Unknown config key: {key}")
    except Exception as e:
        logger.error(f"Failed to set config value {key}: {e}")


def find_config_file() -> Optional[str]:
    """
    Find configuration file in common locations.
    
    Returns:
        Path to configuration file if found, None otherwise
    """
    # Common configuration file names
    config_names = [
        'jupyter_sql_extension.json',
        'jupyter_sql_extension.yaml',
        'jupyter_sql_extension.yml',
        '.jupyter_sql_extension.json',
        '.jupyter_sql_extension.yaml',
        '.jupyter_sql_extension.yml'
    ]
    
    # Common configuration directories
    config_dirs = [
        os.getcwd(),  # Current working directory
        os.path.expanduser('~'),  # Home directory
        os.path.expanduser('~/.config'),  # User config directory
        os.path.expanduser('~/.jupyter'),  # Jupyter config directory
        '/etc/jupyter',  # System config directory
    ]
    
    # Search for configuration files
    for config_dir in config_dirs:
        for config_name in config_names:
            config_path = os.path.join(config_dir, config_name)
            if os.path.exists(config_path):
                return config_path
    
    return None


def create_default_config_file(file_path: str, format: str = 'json') -> None:
    """
    Create a default configuration file.
    
    Args:
        file_path: Path to create the configuration file
        format: File format ('json' or 'yaml')
        
    Raises:
        ConfigurationError: If file creation fails
    """
    try:
        # Get default configuration
        config = ExtensionConfig()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write configuration file
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(config.to_dict(), f, indent=2)
            elif format.lower() in ['yaml', 'yml']:
                yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
        
        logger.info(f"Created default configuration file: {file_path}")
        
    except Exception as e:
        raise ConfigurationError(f"Failed to create configuration file: {e}")


def get_environment_config() -> Dict[str, str]:
    """
    Get all environment variables related to the extension.
    
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    prefix = 'JUPYTER_SQL_EXTENSION_'
    
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()
            env_vars[config_key] = value
    
    return env_vars


def print_config_summary(config: ExtensionConfig) -> None:
    """
    Print a summary of the current configuration.
    
    Args:
        config: Configuration to summarize
    """
    print("Jupyter SQL Extension Configuration Summary:")
    print("=" * 50)
    
    # Service configuration
    print(f"SQL Service URL: {config.sql_service_url}")
    print(f"API Key: {'***' if config.sql_service_api_key else 'Not set'}")
    
    # Connection settings
    print(f"Default Timeout: {config.default_timeout}s")
    print(f"Max Retries: {config.max_retries}")
    print(f"Connection Pool Size: {config.connection_pool_size}")
    
    # Query settings
    print(f"Query Timeout: {config.query_timeout}s")
    print(f"Max Query Length: {config.max_query_length} characters")
    print(f"Allow Write Operations: {config.allow_write_operations}")
    print(f"Allow Non-Select Queries: {config.allow_non_select_queries}")
    
    # Output settings
    print(f"Default Output Format: {config.default_output_format.value}")
    print(f"Max Display Rows: {config.max_display_rows}")
    print(f"Enable Rich Output: {config.enable_rich_output}")
    
    # Security settings
    print(f"Enable SQL Validation: {config.enable_sql_validation}")
    print(f"Enable Input Sanitization: {config.enable_input_sanitization}")
    
    # Logging settings
    print(f"Enable Logging: {config.enable_logging}")
    print(f"Log Level: {config.log_level}")
    print(f"Log File: {config.log_file or 'Not set'}")
    
    # Caching settings
    print(f"Enable Caching: {config.enable_caching}")
    print(f"Cache TTL: {config.cache_ttl}s")
    print(f"Max Cache Size: {config.max_cache_size}")
    
    # Performance settings
    print(f"Enable Metrics: {config.enable_metrics}")
    print(f"Enable Profiling: {config.enable_profiling}")
    print(f"Max Concurrent Queries: {config.max_concurrent_queries}")
    
    print("=" * 50) 