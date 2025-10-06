# SQL Extension for Jupyter

A powerful Jupyter extension designed to work seamlessly with the **SyneHQ.com** data platform, providing secure access to your connected data sources without the need for managing database credentials.

[![Release to PyPI](https://github.com/SyneHQ/jupyter.sql-extension/actions/workflows/release.yml/badge.svg?branch=sudo)](https://github.com/SyneHQ/jupyter.sql-extension/actions/workflows/release.yml)

## About SyneHQ Integration

This extension is specifically built for the [SyneHQ.com](https://synehq.com) data platform, which provides:
- **Zero-Credential Data Access**: Connect to your data sources without exposing database credentials
- **Unified Data Platform**: Access all your connected data sources through a single, secure interface
- **Enterprise-Grade Security**: Built-in authentication, authorization, and audit logging
- **Multi-Platform Support**: Works with your favorite data analysis platforms including Jupyter, R, and more

## Key Features

### 🔐 Secure Connection Management
- **Credential-Free Access**: Retrieve database connections securely through SyneHQ's internal services
- **Enterprise Authentication**: Built-in support for SSO, OAuth, and enterprise identity providers
- **Connection Pooling**: Efficient connection management with automatic retry and failover

### 🛡️ Security & Validation
- **SQL Injection Prevention**: Advanced input validation and query sanitization
- **Query Safety Checks**: Automatic detection of potentially harmful operations
- **Audit Logging**: Complete query execution tracking for compliance and monitoring

### 📊 Rich Output Formatting
- **Pandas DataFrames**: Native support for DataFrame output with automatic type inference
- **Interactive Tables**: HTML tables with sorting, filtering, and pagination
- **JSON Export**: Structured data output for API integrations
- **Custom Visualization**: Support for charts and graphs integration

### 🔄 Advanced Query Features
- **Variable Assignment**: Assign query results to Python variables using intuitive syntax
- **Python Variable Substitution**: Use Python variables, expressions, and function calls directly in SQL queries
- **Type-Safe Formatting**: Automatic type detection and SQL-safe formatting for all Python data types
- **Expression Evaluation**: Evaluate complex Python expressions safely within SQL queries
- **Async Execution**: Non-blocking query execution for better performance
- **Query Caching**: Intelligent caching to reduce redundant database calls

### 📈 Performance & Monitoring
- **Execution Metrics**: Detailed performance tracking and query optimization insights
- **Connection Health**: Real-time monitoring of database connection status
- **Error Recovery**: Automatic retry mechanisms with exponential backoff

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Access to SyneHQ.com data platform

### Install via pip
```bash
pip install syne-sql-extension
```

### Install from source
```bash
git clone https://github.com/synehq/jupyter-sql-extension.git
cd jupyter-sql-extension
pip install -e .
```

### Load the extension in Jupyter
```python
%load_ext syne_sql_extension
```

## Quick Start

### 1. Set up Authentication
```python
# Option 1: Set global variable (recommended for Jupyter notebooks)
SYNE_OAUTH_KEY = 'your_api_key_here'

# Option 2: Use environment variable
# export SYNE_OAUTH_KEY='your_api_key_here'

# Option 3: Provide via command line (most explicit)
%%sql my_database --api-key your_api_key_here
SELECT * FROM users LIMIT 10
```

### 2. Connect to SyneHQ
```python
%%sql my_database
SELECT * FROM users LIMIT 10
```

### 3. Use with variables
```python
# Assign results to a variable
%%sql analytics_db --output users_df
SELECT user_id, name, email, created_at 
FROM users 
WHERE created_at >= '2024-01-01'
```

### 4. Parameterized queries
```python
user_limit = 100
department = 'engineering'

%%sql hr_db
SELECT * FROM employees 
WHERE department = {department} 
LIMIT {user_limit}
```

### 5. Different output formats
```python
# DataFrame output (default)
%%sql sales_db --format dataframe
SELECT product, SUM(revenue) as total_revenue 
FROM sales 
GROUP BY product

# HTML table
%%sql sales_db --format html
SELECT * FROM products WHERE price > 100

# JSON output
%%sql api_db --format json
SELECT config FROM settings WHERE active = true
```

## Authentication

The extension supports multiple ways to provide your SyneHQ API key for authentication. The API key is resolved in the following order of preference:

### 1. Command Line (Most Explicit)
```python
%%sql my_db --api-key your_api_key_here
SELECT * FROM users LIMIT 10
```

### 2. Global Variable (Recommended for Jupyter)
```python
# Set once at the beginning of your notebook
SYNE_OAUTH_KEY = 'your_api_key_here'

# Then use without specifying the key
%%sql my_db
SELECT * FROM users LIMIT 10
```

### 3. Environment Variable
```bash
# Set in your shell environment
export SYNE_OAUTH_KEY='your_api_key_here'

# Or in your Jupyter environment
import os
os.environ['SYNE_OAUTH_KEY'] = 'your_api_key_here'
```

### Security Best Practices

- **Never hardcode API keys** in your notebooks or commit them to version control
- **Use environment variables** for production deployments
- **Use global variables** for interactive development in Jupyter
- **Rotate API keys regularly** for enhanced security

### Getting Your API Key

1. Log in to your [SyneHQ account](https://synehq.com)
2. Navigate to Teams > Choose the team >API Keys
3. Generate a new API key with appropriate permissions
4. Copy the key and use one of the authentication methods above

## Usage Examples

### Basic Queries
```python
# Simple select
%%sql main_db
SELECT COUNT(*) as total_users FROM users

# Join multiple tables
%%sql warehouse
SELECT 
    u.name,
    p.product_name,
    o.order_date,
    o.amount
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN products p ON o.product_id = p.id
WHERE o.order_date >= '2024-01-01'
```

### Data Analysis Workflow
```python

# Load data into DataFrame
%%sql analytics
sales_data >> SELECT 
    DATE(order_date) as date,
    product_category,
    SUM(amount) as daily_revenue,
    COUNT(*) as order_count
FROM orders 
WHERE order_date >= '2024-01-01'
GROUP BY DATE(order_date), product_category
ORDER BY date DESC

# Analyze the data
print(f"Total revenue: ${sales_data['daily_revenue'].sum():,.2f}")
print(f"Average daily orders: {sales_data['order_count'].mean():.1f}")

# Create visualization
sales_data.groupby('product_category')['daily_revenue'].sum().plot(kind='bar')
```

### Advanced Features
```python
# Using Python variables in queries with enhanced syntax
start_date = '2024-01-01'
end_date = '2024-12-31'
min_revenue = 1000
user_ids = [1, 2, 3, 4, 5]

# Simple variable substitution
%%sql finance
SELECT 
    customer_id,
    SUM(amount) as total_spent
FROM transactions 
WHERE transaction_date BETWEEN {start_date} AND {end_date}
GROUP BY customer_id
HAVING SUM(amount) >= {min_revenue}
ORDER BY total_spent DESC

# List variables with automatic formatting
%%sql analytics
SELECT * FROM users WHERE id IN {user_ids}

# Type-specific formatting
%%sql analytics
SELECT * FROM users WHERE id IN {user_ids:list}

# Expression evaluation
%%sql finance
SELECT * FROM products WHERE price = {min_revenue * 1.5}

# Complex expressions with functions
from datetime import datetime, timedelta
%%sql analytics
SELECT * FROM users WHERE created_at >= {datetime.now() - timedelta(days=30)}
```

## Python Variable Support

The extension provides comprehensive Python variable substitution in SQL queries with multiple syntax options and safety features.

### Syntax Options

#### 1. Simple Variable Substitution
```python
user_id = 123
user_name = "John Doe"

%%sql my_connection -k my_key
SELECT * FROM users WHERE id = {user_id}
SELECT * FROM users WHERE name = {user_name}
```

#### 2. Type-Specific Formatting
```python
user_ids = [1, 2, 3, 4, 5]
price = 99.99
created_date = datetime(2024, 1, 1)

%%sql my_connection -k my_key
SELECT * FROM users WHERE id IN {user_ids:list}
SELECT * FROM products WHERE price = {price:number}
SELECT * FROM users WHERE created_at >= {created_date:date}
```

#### 3. Expression Evaluation
```python
base_price = 100
discount_rate = 0.1
tax_rate = 0.08

%%sql my_connection -k my_key
SELECT * FROM products WHERE final_price = {base_price * (1 - discount_rate) * (1 + tax_rate)}
```

#### 4. Function Calls and Complex Expressions
```python
from datetime import datetime, timedelta

%%sql my_connection -k my_key
SELECT * FROM users WHERE created_at >= {datetime.now() - timedelta(days=30)}
SELECT * FROM products WHERE rounded_price = {round(99.99 * 1.15, 2)}
```

### Supported Data Types

- **Strings**: Automatically quoted and escaped
- **Numbers**: Used as-is without quotes
- **Lists/Tuples**: Formatted as SQL IN clauses
- **Booleans**: Converted to strings
- **None**: Converted to SQL NULL
- **Datetime objects**: Formatted as ISO strings

### Security Features

- **Safe Expression Evaluation**: Only safe built-in functions are allowed
- **Pattern Blocking**: Dangerous patterns like `import`, `exec`, `eval` are blocked
- **Function Blacklisting**: Dangerous functions like `os`, `sys`, `subprocess` are blocked
- **Sandboxed Environment**: Expressions run in a restricted environment

### Complete Example

```python
# Set up variables
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
min_age = 18
max_age = 65
active_statuses = ['active', 'premium']
excluded_users = [999, 1000, 1001]

# Complex query with multiple variable types
%%sql analytics_db -k my_key
SELECT 
    u.id,
    u.name,
    u.email,
    u.age,
    u.status,
    COUNT(o.id) as order_count,
    SUM(o.total) as total_spent
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.created_at BETWEEN {start_date:date} AND {end_date:date}
  AND u.age BETWEEN {min_age} AND {max_age}
  AND u.status IN {active_statuses:list}
  AND u.id NOT IN {excluded_users:list}
GROUP BY u.id, u.name, u.email, u.age, u.status
HAVING COUNT(o.id) > 0
ORDER BY total_spent DESC
LIMIT 100
```

For detailed documentation on Python variable support, see [PYTHON_VARIABLE_SUPPORT.md](PYTHON_VARIABLE_SUPPORT.md).

## Connection Management

### Available Connections
```python
# List available connections
%%sql --list-connections -k {key}
```

### Test Connection
```python
# Test if connection is working
%%sql {connection_id}
SELECT 1
```

## Error Handling

The extension provides comprehensive error handling with user-friendly messages:

```python
%%sql invalid_db
SELECT * FROM nonexistent_table
```

Common error scenarios:
- **Connection Errors**: Invalid connection ID, network issues, authentication failures
- **Query Errors**: SQL syntax errors, table not found, permission denied
- **Validation Errors**: SQL injection attempts, unsafe operations
- **Timeout Errors**: Long-running queries, connection timeouts

## Security Features

### SQL Injection Prevention
```python
# ❌ This will be blocked
user_input = "'; DROP TABLE users; --"
%%sql db
SELECT * FROM users WHERE name = '{user_input}'

# ✅ Use parameter binding instead
user_input = "John Doe"
%%sql db
SELECT * FROM users WHERE name = {user_input}
```

### Query Validation
The extension automatically validates queries for:
- Potentially dangerous operations (DROP, DELETE, etc.)
- SQL injection patterns
- Syntax errors
- Resource usage limits

## Performance Optimization

### Query Caching
```python
# Enable caching for repeated queries
%%sql db --cache
SELECT expensive_aggregation() FROM large_table
```

### Async Execution
```python
# Run multiple queries concurrently
import asyncio

async def run_queries():
    tasks = []
    for db in ['db1', 'db2', 'db3']:
        task = execute_query(f"%%sql {db}\nSELECT COUNT(*) FROM table")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

## Troubleshooting

### Common Issues

**Extension not loading:**
```python
# Check if extension is properly installed
%load_ext syne_sql_extension
```

**Connection failures:**
- Verify your SyneHQ API credentials
- Check network connectivity to SyneHQ services
- Ensure your workspace has access to the requested data sources

**Query errors:**
- Validate SQL syntax
- Check table and column names
- Verify permissions for the data source

### Debug Mode
```python
# Enable debug logging
%%sql db --debug
SELECT * FROM users
```

### Getting Help
- Check the [SyneHQ Documentation](https://docs.synehq.com)
- Visit our [GitHub Issues](https://github.com/synehq/jupyter-sql-extension/issues)
- Contact support at [support@synehq.com](mailto:support@synehq.com)

## API Reference

### Magic Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--connection-id` | SyneHQ connection identifier | Required |
| `--api-key` | SyneHQ API key for authentication | Auto-detected |
| `--output` | Variable name for query results | None |
| `--format` | Output format (dataframe, html, json) | dataframe |
| `--timeout` | Query timeout in seconds | 30 |
| `--cache` | Enable query caching | false |
| `--debug` | Enable debug logging | false |
| `--test` | Test connection without executing query | false |

### Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `api_url` | SyneHQ API endpoint | https://api.synehq.com |
| `timeout` | Default query timeout | 30 |
| `retry_attempts` | Number of retry attempts | 3 |
| `cache_enabled` | Enable query caching | true |
| `cache_ttl` | Cache time-to-live (seconds) | 300 |
| `output_format` | Default output format | dataframe |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/synehq/jupyter-sql-extension.git
cd jupyter-sql-extension
pip install -e ".[dev]"
pre-commit install
```

### Running Tests
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs.synehq.com](https://docs.synehq.com)
- **Issues**: [GitHub Issues](https://github.com/synehq/jupyter.sql-extension/issues)
- **Email**: [support@synehq.com](mailto:support@synehq.com)
- **SyneHQ Platform**: [synehq.com](https://synehq.com)

---

**Made with ❤️ by the SyneHQ team**
