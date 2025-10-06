# Python Variable Support in SQL Queries

The Jupyter SQL Extension now supports comprehensive Python variable substitution in SQL queries, allowing you to dynamically inject Python values, expressions, and function calls directly into your SQL statements.

## Features

### 1. Simple Variable Substitution
Use `{variable_name}` to substitute Python variables directly into your SQL queries.

```python
# Define variables
user_id = 123
user_name = "John Doe"
is_active = True

# Use in SQL queries
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id = {user_id}

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE name = {user_name}

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE active = {is_active}
```

### 2. Type-Specific Formatting
Use `{variable_name:type}` to explicitly control how variables are formatted in SQL.

#### Supported Types:
- `string` - Wraps value in single quotes and escapes special characters
- `number` - Uses value as-is (no quotes)
- `list` - Formats as SQL IN clause: `(value1, value2, value3)`
- `date` - Formats datetime objects as ISO strings
- `raw` - Uses value as-is without any formatting (use with caution)

```python
# Type-specific examples
user_ids = [1, 2, 3, 4, 5]
user_names = ["Alice", "Bob", "Charlie"]
created_date = datetime(2024, 1, 1)
price = 99.99

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id IN {user_ids:list}

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE name IN {user_names:list}

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE created_at >= {created_date:date}

%%sqlconnect my_connection --api-key my_key
SELECT * FROM products WHERE price = {price:number}
```

### 3. Expression Evaluation
Use `{expression}` to evaluate Python expressions directly in your SQL queries.

```python
# Mathematical expressions
base_price = 100
discount_rate = 0.1
tax_rate = 0.08

%%sqlconnect my_connection --api-key my_key
SELECT * FROM products WHERE final_price = {base_price * (1 - discount_rate) * (1 + tax_rate)}

# String operations
first_name = "John"
last_name = "Doe"

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE full_name = {first_name + " " + last_name}

# List operations
items = [1, 2, 3, 4, 5]

%%sqlconnect my_connection --api-key my_key
SELECT * FROM orders WHERE item_count = {len(items)}

# Range generation
min_age = 18
max_age = 65

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE age IN {list(range(min_age, max_age + 1, 10))}
```

### 4. Function Calls and Complex Expressions
Use function calls and complex expressions within `{...}` blocks.

```python
from datetime import datetime, timedelta

# Date calculations
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE created_at >= {datetime.now() - timedelta(days=30)}

# Mathematical functions
%%sqlconnect my_connection --api-key my_key
SELECT * FROM products WHERE rounded_price = {round(99.99 * 1.15, 2)}

# Conditional expressions
user_count = 5

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE status = {("active" if user_count > 0 else "inactive")}

# List comprehensions
numbers = [1, 2, 3, 4, 5]

%%sqlconnect my_connection --api-key my_key
SELECT * FROM products WHERE id IN {[x * 2 for x in numbers]}
```

## Data Type Handling

### Automatic Type Detection
When no type is specified, the extension automatically detects the appropriate SQL formatting:

- **Strings**: Wrapped in single quotes with escaped special characters
- **Numbers**: Used as-is without quotes
- **Lists/Tuples**: Formatted as SQL IN clauses
- **Booleans**: Converted to strings
- **None**: Converted to SQL NULL
- **Datetime objects**: Formatted as ISO strings

### Special Character Handling
Strings with special characters are automatically escaped:

```python
description = "It's a great product!"

%%sqlconnect my_connection --api-key my_key
SELECT * FROM products WHERE description = {description}
# Results in: SELECT * FROM products WHERE description = 'It''s a great product!'
```

### Empty Values
Empty lists and None values are handled appropriately:

```python
empty_list = []
none_value = None

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id IN {empty_list}
# Results in: SELECT * FROM users WHERE id IN ()

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE deleted_at = {none_value}
# Results in: SELECT * FROM users WHERE deleted_at = NULL
```

## Security Features

### Safe Expression Evaluation
The extension includes comprehensive security measures to prevent code injection:

- **Restricted Built-ins**: Only safe built-in functions are available
- **Pattern Blocking**: Dangerous patterns like `import`, `exec`, `eval` are blocked
- **Function Blacklisting**: Dangerous functions like `os`, `sys`, `subprocess` are blocked
- **Sandboxed Environment**: Expressions run in a restricted environment

### Allowed Operations
Safe operations include:
- Mathematical operations (`+`, `-`, `*`, `/`, `%`, `**`)
- Comparison operations (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- Logical operations (`and`, `or`, `not`)
- Built-in functions (`len`, `str`, `int`, `float`, `list`, `tuple`, `dict`, `set`)
- Type checking (`isinstance`, `issubclass`, `type`)
- String operations (`+`, `*`, slicing, methods)
- List operations (`append`, `extend`, `index`, `count`, etc.)

### Blocked Operations
Unsafe operations that are blocked:
- Import statements (`import`, `from`)
- Code execution (`exec`, `eval`, `compile`)
- File operations (`open`, `file`)
- System access (`os`, `sys`, `subprocess`)
- Network operations (`socket`, `urllib`, `requests`)
- Reflection (`getattr`, `setattr`, `hasattr`, `dir`, `vars`)
- Global/local access (`globals`, `locals`)

## Error Handling

### Variable Not Found
If a variable is not found in the local namespace, a `ValidationError` is raised:

```python
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id = {missing_variable}
# Raises: ValidationError: Variable 'missing_variable' not found in local namespace
```

### Invalid Expressions
If an expression cannot be evaluated or contains unsafe code, a `ValidationError` is raised:

```python
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id = {user_id +}
# Raises: ValidationError: Expression evaluation failed: invalid syntax

%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id = {import os}
# Raises: ValidationError: Unsafe expression detected: import\s+
```

## Best Practices

### 1. Use Type-Specific Formatting
When you know the expected SQL type, use explicit type formatting:

```python
# Good
user_ids = [1, 2, 3]
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id IN {user_ids:list}

# Also good (auto-detection works)
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id IN {user_ids}
```

### 2. Validate Variables Before Use
Check that variables exist and have expected values:

```python
if 'user_id' in locals() and user_id is not None:
    %%sqlconnect my_connection --api-key my_key
    SELECT * FROM users WHERE id = {user_id}
```

### 3. Use Expressions for Dynamic Values
For calculated values, use expressions instead of pre-calculating:

```python
# Good - dynamic calculation
base_price = 100
discount = 0.1
%%sqlconnect my_connection --api-key my_key
SELECT * FROM products WHERE final_price = {base_price * (1 - discount)}

# Less ideal - pre-calculated
final_price = base_price * (1 - discount)
%%sqlconnect my_connection --api-key my_key
SELECT * FROM products WHERE final_price = {final_price}
```

### 4. Handle Edge Cases
Consider edge cases like empty lists and None values:

```python
# Safe handling of potentially empty lists
user_ids = get_user_ids()  # Might return empty list
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id IN {user_ids if user_ids else [0]}
```

## Examples

### Complete Example: User Analytics Dashboard

```python
# Set up variables
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
min_age = 18
max_age = 65
active_statuses = ['active', 'premium']
excluded_users = [999, 1000, 1001]

# Complex query with multiple variable types
%%sqlconnect analytics_db --api-key my_key
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

### Example: Dynamic Report Generation

```python
# Dynamic report parameters
report_type = "monthly"
current_month = datetime.now().month
current_year = datetime.now().year
department_ids = [1, 2, 3, 4, 5]
min_salary = 50000
max_salary = 150000

# Generate different queries based on parameters
if report_type == "monthly":
    %%sqlconnect hr_db --api-key my_key
    SELECT 
        d.name as department,
        COUNT(e.id) as employee_count,
        AVG(e.salary) as avg_salary,
        MIN(e.salary) as min_salary,
        MAX(e.salary) as max_salary
    FROM employees e
    JOIN departments d ON e.department_id = d.id
    WHERE e.department_id IN {department_ids:list}
      AND e.salary BETWEEN {min_salary} AND {max_salary}
      AND MONTH(e.hire_date) = {current_month}
      AND YEAR(e.hire_date) = {current_year}
    GROUP BY d.id, d.name
    ORDER BY avg_salary DESC
```

## Troubleshooting

### Common Issues

1. **Variable not found**: Ensure the variable is defined in the current namespace
2. **Invalid expression**: Check for syntax errors in complex expressions
3. **Unsafe expression**: Remove dangerous functions or patterns
4. **Type mismatch**: Use explicit type formatting when auto-detection fails

### Debug Mode
Use verbose mode to see variable substitution details:

```python
%%sqlconnect my_connection --api-key my_key --verbose
SELECT * FROM users WHERE id = {user_id}
```

This will show:
- Variables found and substituted
- Expressions evaluated
- Final SQL query generated

## Migration from Previous Versions

If you were using the previous simple `{variable_name}` syntax, your existing code will continue to work without changes. The new features are additive and backward-compatible.

### Upgrading Your Code

You can gradually adopt new features:

```python
# Old way (still works)
user_id = 123
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id = {user_id}

# New way with explicit typing
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id = {user_id:number}

# New way with expressions
%%sqlconnect my_connection --api-key my_key
SELECT * FROM users WHERE id = {user_id * 2}
```

This comprehensive Python variable support makes the Jupyter SQL Extension much more powerful and flexible for dynamic SQL query generation while maintaining security and ease of use.
