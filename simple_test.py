#!/usr/bin/env python3
"""
Simple test script for Python variable support functionality.

This script tests the core variable substitution logic without requiring IPython.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the extension directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'syne_sql_extension'))

# Import only the core functionality we need
from syne_sql_extension.exceptions import ValidationError


class MockSQLConnectMagic:
    """Mock version of SQLConnectMagic for testing variable substitution."""
    
    def _substitute_variables(self, query: str, local_ns: Dict[str, Any], verbose: bool = False) -> str:
        """
        Substitute Python variables in the SQL query safely.

        Supports multiple syntax patterns:
        1. {variable_name} - Simple variable substitution
        2. {variable_name:type} - Type-specific formatting
        3. {expression} - Python expression evaluation
        4. {function_call()} - Function call evaluation
        """
        import re
        
        # Pattern 1: Simple variable substitution {variable_name}
        simple_pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        simple_matches = re.findall(simple_pattern, query)
        
        # Pattern 2: Type-specific formatting {variable_name:type}
        typed_pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)\}'
        typed_matches = re.findall(typed_pattern, query)
        
        # Pattern 3: Expression evaluation {expression}
        expression_pattern = r'\{([^}]+)\}'
        expression_matches = re.findall(expression_pattern, query)
        
        # Filter out simple and typed matches from expression matches
        expression_matches = [expr for expr in expression_matches 
                            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', expr) and
                               not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z_][a-zA-Z0-9_]*$', expr)]

        all_variables = set(simple_matches + [match[0] for match in typed_matches])
        
        if all_variables and verbose:
            print(f"🔄 Found variables to substitute: {sorted(all_variables)}")
        
        if expression_matches and verbose:
            print(f"🔄 Found expressions to evaluate: {expression_matches}")

        # Process simple variable substitutions
        for var_name in simple_matches:
            if var_name in local_ns:
                var_value = local_ns[var_name]
                safe_value = self._format_value_for_sql(var_value, 'auto', verbose)
                query = query.replace(f"{{{var_name}}}", safe_value)
                
                if verbose:
                    print(f"🔄 Substituted {var_name} = {safe_value}")
            else:
                raise ValidationError(f"Variable '{var_name}' not found in local namespace")

        # Process typed variable substitutions
        for var_name, var_type in typed_matches:
            if var_name in local_ns:
                var_value = local_ns[var_name]
                safe_value = self._format_value_for_sql(var_value, var_type, verbose)
                query = query.replace(f"{{{var_name}:{var_type}}}", safe_value)
                
                if verbose:
                    print(f"🔄 Substituted {var_name}:{var_type} = {safe_value}")
            else:
                raise ValidationError(f"Variable '{var_name}' not found in local namespace")

        # Process expression evaluations
        for expression in expression_matches:
            try:
                # Evaluate the expression safely
                result = self._evaluate_expression(expression, local_ns, verbose)
                safe_value = self._format_value_for_sql(result, 'auto', verbose)
                query = query.replace(f"{{{expression}}}", safe_value)
                
                if verbose:
                    print(f"🔄 Evaluated expression '{expression}' = {safe_value}")
            except Exception as e:
                raise ValidationError(f"Failed to evaluate expression '{expression}': {e}")

        return query

    def _format_value_for_sql(self, value: Any, format_type: str = 'auto', verbose: bool = False) -> str:
        """Format a Python value for safe use in SQL queries."""
        if value is None:
            return 'NULL'
        
        if format_type == 'auto':
            # Auto-detect format based on value type
            if isinstance(value, str):
                format_type = 'string'
            elif isinstance(value, (int, float)):
                format_type = 'number'
            elif isinstance(value, (list, tuple)):
                format_type = 'list'
            elif isinstance(value, datetime):
                format_type = 'date'
            else:
                format_type = 'string'
        
        if format_type == 'string':
            # Escape single quotes and wrap in quotes
            escaped_value = str(value).replace("'", "''")
            return f"'{escaped_value}'"
        
        elif format_type == 'number':
            return str(value)
        
        elif format_type == 'list':
            # Convert to SQL IN clause format
            if not value:
                return '()'
            
            safe_items = []
            for item in value:
                if isinstance(item, str):
                    escaped_item = str(item).replace("'", "''")
                    safe_items.append(f"'{escaped_item}'")
                elif isinstance(item, (int, float)):
                    safe_items.append(str(item))
                elif item is None:
                    safe_items.append('NULL')
                else:
                    escaped_item = str(item).replace("'", "''")
                    safe_items.append(f"'{escaped_item}'")
            
            return f"({', '.join(safe_items)})"
        
        elif format_type == 'date':
            if isinstance(value, datetime):
                return f"'{value.isoformat()}'"
            else:
                return f"'{str(value)}'"
        
        elif format_type == 'raw':
            # Use value as-is (be careful with this!)
            return str(value)
        
        else:
            # Default to string formatting
            escaped_value = str(value).replace("'", "''")
            return f"'{escaped_value}'"

    def _evaluate_expression(self, expression: str, local_ns: Dict[str, Any], verbose: bool = False) -> Any:
        """Safely evaluate a Python expression in the given namespace."""
        import re
        
        # Security check - only allow safe operations
        unsafe_patterns = [
            r'import\s+',
            r'from\s+',
            r'__\w+__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'compile\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'hasattr\s*\(',
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, expression, re.IGNORECASE):
                raise ValidationError(f"Unsafe expression detected: {pattern}")
        
        # Additional safety: check for dangerous function calls
        dangerous_functions = [
            'os', 'sys', 'subprocess', 'shutil', 'pickle', 'marshal',
            'socket', 'urllib', 'requests', 'http', 'ftplib', 'smtplib'
        ]
        
        for func in dangerous_functions:
            if func in expression:
                raise ValidationError(f"Dangerous function '{func}' detected in expression")
        
        try:
            # Create a safe evaluation environment with only safe built-ins
            safe_globals = {
                '__builtins__': {
                    # Safe built-in functions
                    'abs': abs, 'all': all, 'any': any, 'bin': bin, 'bool': bool,
                    'chr': chr, 'dict': dict, 'divmod': divmod, 'enumerate': enumerate,
                    'filter': filter, 'float': float, 'format': format, 'hex': hex,
                    'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
                    'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max,
                    'min': min, 'oct': oct, 'ord': ord, 'pow': pow, 'range': range,
                    'repr': repr, 'reversed': reversed, 'round': round, 'set': set,
                    'sorted': sorted, 'str': str, 'sum': sum, 'tuple': tuple,
                    'type': type, 'zip': zip, 'frozenset': frozenset,
                    # Safe constants
                    'None': None, 'True': True, 'False': False
                }
            }
            
            # Evaluate the expression
            result = eval(expression, safe_globals, local_ns)
            
            if verbose:
                print(f"🔄 Expression '{expression}' evaluated to: {type(result).__name__}")
            
            return result
            
        except Exception as e:
            raise ValidationError(f"Expression evaluation failed: {e}")


def test_variable_substitution():
    """Test various variable substitution patterns."""
    
    print("🧪 Testing Python Variable Support in SQL Queries")
    print("=" * 60)
    
    # Create a mock magic instance for testing
    magic = MockSQLConnectMagic()
    
    # Test data
    test_variables = {
        'user_id': 123,
        'user_name': 'John Doe',
        'user_ids': [1, 2, 3, 4, 5],
        'user_names': ['Alice', 'Bob', 'Charlie'],
        'min_age': 18,
        'max_age': 65,
        'is_active': True,
        'created_date': datetime(2024, 1, 1),
        'days_ago': 30,
        'price': 99.99,
        'description': "It's a great product!",
        'empty_list': [],
        'none_value': None
    }
    
    # Test cases
    test_cases = [
        {
            'name': 'Simple Variable Substitution',
            'query': 'SELECT * FROM users WHERE id = {user_id}',
            'expected': "SELECT * FROM users WHERE id = 123"
        },
        {
            'name': 'String Variable with Quotes',
            'query': "SELECT * FROM users WHERE name = {user_name}",
            'expected': "SELECT * FROM users WHERE name = 'John Doe'"
        },
        {
            'name': 'List Variable (Auto-detection)',
            'query': 'SELECT * FROM users WHERE id IN {user_ids}',
            'expected': "SELECT * FROM users WHERE id IN (1, 2, 3, 4, 5)"
        },
        {
            'name': 'Type-specific Formatting (List)',
            'query': 'SELECT * FROM users WHERE id IN {user_ids:list}',
            'expected': "SELECT * FROM users WHERE id IN (1, 2, 3, 4, 5)"
        },
        {
            'name': 'Type-specific Formatting (String)',
            'query': 'SELECT * FROM users WHERE name = {user_name:string}',
            'expected': "SELECT * FROM users WHERE name = 'John Doe'"
        },
        {
            'name': 'Type-specific Formatting (Number)',
            'query': 'SELECT * FROM users WHERE age = {min_age:number}',
            'expected': 'SELECT * FROM users WHERE age = 18'
        },
        {
            'name': 'Boolean Variable',
            'query': 'SELECT * FROM users WHERE active = {is_active}',
            'expected': "SELECT * FROM users WHERE active = True"
        },
        {
            'name': 'Date Variable',
            'query': 'SELECT * FROM users WHERE created_at >= {created_date:date}',
            'expected': "SELECT * FROM users WHERE created_at >= '2024-01-01T00:00:00'"
        },
        {
            'name': 'Float Variable',
            'query': 'SELECT * FROM users WHERE price = {price}',
            'expected': 'SELECT * FROM users WHERE price = 99.99'
        },
        {
            'name': 'String with Special Characters',
            'query': 'SELECT * FROM products WHERE description = {description}',
            'expected': "SELECT * FROM products WHERE description = 'It''s a great product!'"
        },
        {
            'name': 'Empty List',
            'query': 'SELECT * FROM users WHERE id IN {empty_list}',
            'expected': 'SELECT * FROM users WHERE id IN ()'
        },
        {
            'name': 'None Value',
            'query': 'SELECT * FROM users WHERE deleted_at = {none_value}',
            'expected': 'SELECT * FROM users WHERE deleted_at = NULL'
        }
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for test_case in test_cases:
        try:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"   Query: {test_case['query']}")
            
            # Test the variable substitution
            result = magic._substitute_variables(
                test_case['query'], 
                test_variables, 
                verbose=False
            )
            
            print(f"   Result: {result}")
            print(f"   Expected: {test_case['expected']}")
            
            if result == test_case['expected']:
                print("   ✅ PASSED")
                passed += 1
            else:
                print("   ❌ FAILED")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return passed, failed


def test_expression_evaluation():
    """Test expression evaluation in SQL queries."""
    
    print("\n🧪 Testing Expression Evaluation in SQL Queries")
    print("=" * 60)
    
    magic = MockSQLConnectMagic()
    
    # Test variables for expressions
    test_variables = {
        'min_age': 18,
        'max_age': 65,
        'base_price': 100,
        'discount_rate': 0.1,
        'tax_rate': 0.08,
        'user_count': 5,
        'items': [1, 2, 3, 4, 5],
        'name': 'John',
        'surname': 'Doe'
    }
    
    # Test cases for expressions
    expression_tests = [
        {
            'name': 'Simple Arithmetic',
            'query': 'SELECT * FROM products WHERE price = {base_price * 1.1}',
            'expected': 'SELECT * FROM products WHERE price = 110.00000000000001'
        },
        {
            'name': 'Complex Calculation',
            'query': 'SELECT * FROM products WHERE final_price = {base_price * (1 - discount_rate) * (1 + tax_rate)}',
            'expected': 'SELECT * FROM products WHERE final_price = 97.2'
        },
        {
            'name': 'String Concatenation',
            'query': 'SELECT * FROM users WHERE full_name = {name + " " + surname}',
            'expected': 'SELECT * FROM users WHERE full_name = \'John Doe\''
        },
        {
            'name': 'List Length',
            'query': 'SELECT * FROM orders WHERE item_count = {len(items)}',
            'expected': 'SELECT * FROM orders WHERE item_count = 5'
        },
        {
            'name': 'Range Generation',
            'query': 'SELECT * FROM users WHERE age IN {list(range(min_age, max_age + 1, 10))}',
            'expected': "SELECT * FROM users WHERE age IN (18, 28, 38, 48, 58)"
        },
        {
            'name': 'Conditional Expression',
            'query': 'SELECT * FROM users WHERE status = {("active" if user_count > 0 else "inactive")}',
            'expected': "SELECT * FROM users WHERE status = 'active'"
        },
        {
            'name': 'Mathematical Functions',
            'query': 'SELECT * FROM products WHERE rounded_price = {round(base_price * 1.15, 2)}',
            'expected': 'SELECT * FROM products WHERE rounded_price = 115.0'
        }
    ]
    
    passed = 0
    failed = 0
    
    for test_case in expression_tests:
        try:
            print(f"\n🔍 Testing: {test_case['name']}")
            print(f"   Query: {test_case['query']}")
            
            result = magic._substitute_variables(
                test_case['query'], 
                test_variables, 
                verbose=False
            )
            
            print(f"   Result: {result}")
            print(f"   Expected: {test_case['expected']}")
            
            if result == test_case['expected']:
                print("   ✅ PASSED")
                passed += 1
            else:
                print("   ❌ FAILED")
                failed += 1
                
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            failed += 1
    
    print(f"\n📊 Expression Test Results: {passed} passed, {failed} failed")
    return passed, failed


def main():
    """Run all tests."""
    
    print("🚀 Starting Python Variable Support Tests")
    print("=" * 80)
    
    total_passed = 0
    total_failed = 0
    
    # Run all test suites
    try:
        passed, failed = test_variable_substitution()
        total_passed += passed
        total_failed += failed
    except Exception as e:
        print(f"❌ Variable substitution tests failed: {e}")
        total_failed += 1
    
    try:
        passed, failed = test_expression_evaluation()
        total_passed += passed
        total_failed += failed
    except Exception as e:
        print(f"❌ Expression evaluation tests failed: {e}")
        total_failed += 1
    
    # Final results
    print("\n" + "=" * 80)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 80)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {(total_passed / (total_passed + total_failed)) * 100:.1f}%")
    
    if total_failed == 0:
        print("🎉 All tests passed! Python variable support is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
