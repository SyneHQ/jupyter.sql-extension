#!/usr/bin/env python3
"""
Setup configuration for Jupyter SQL Extension

A robust Jupyter Notebook extension for executing SQL queries through
internal service connections with security and validation.
"""
from pathlib import Path
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    """Extract version from package __init__.py"""
    version_file = Path(__file__).parent / "syne_sql_extension" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('"')[1]
    return "0.1.0"

# Read long description from README
def get_long_description():
    """Read long description from README file"""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Define package requirements
INSTALL_REQUIRES = [
    # Core Jupyter dependencies
    "jupyter>=1.0.0",
    "ipython>=8.0.0",
    "jupyter-client>=7.0.0",
    "notebook>=6.4.0",

    # HTTP client and async support
    "httpx>=0.24.0",
    "aiohttp>=3.8.0",
    "requests>=2.28.0",

    # Data handling and validation
    "pandas>=1.5.0",
    "pydantic>=2.0.0",
    "marshmallow>=3.19.0",
    "jsonschema>=4.17.0",

    # Security and authentication
    "cryptography>=40.0.0",
    "pyjwt>=2.6.0",
    "keyring>=23.13.0",

    # Configuration and environment
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "toml>=0.10.2",

    # Type checking and validation
    "typing-extensions>=4.5.0",
    "mypy-extensions>=1.0.0",

    # Rich output and formatting
    "rich>=13.0.0",
    "tabulate>=0.9.0",
    "jinja2>=3.1.0",

    # SQL parsing and formatting
    "sqlparse>=0.4.3",
    "sqlalchemy>=2.0.0",  # For SQL validation and parsing

    # Async utilities
    "asyncio-mqtt>=0.13.0",  # If using MQTT for real-time updates
    "websockets>=11.0.0",    # For WebSocket connections

    # Error tracking and monitoring
    "sentry-sdk>=1.20.0",

    # Caching
    "cachetools>=5.3.0",
    "diskcache>=5.6.0",
]

# Development dependencies
EXTRAS_REQUIRE = {
    "dev": [
        # Testing
        "pytest>=7.2.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "pytest-xdist>=3.2.0",
        "pytest-timeout>=2.1.0",
        "factory-boy>=3.2.0",
        "faker>=18.4.0",

        # Code quality
        "black>=23.1.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "pylint>=2.16.0",
        "mypy>=1.1.0",
        "bandit>=1.7.4",
        "safety>=2.3.0",

        # Documentation
        "sphinx>=6.1.0",
        "sphinx-rtd-theme>=1.2.0",
        "sphinx-autodoc-typehints>=1.22.0",
        "myst-parser>=1.0.0",

        # Pre-commit hooks
        "pre-commit>=3.1.0",

        # Build tools
        "build>=0.10.0",
        "twine>=4.0.0",
        "wheel>=0.40.0",

        # Jupyter development
        "jupyter-packaging>=0.12.0",
        "jupyterlab>=4.0.0",
        "notebook>=7.0.0",

        # Performance profiling
        "line-profiler>=4.0.0",
        "memory-profiler>=0.60.0",
    ],

    "grpc": [
        # gRPC support for service communication
        "grpcio>=1.51.0",
        "grpcio-tools>=1.51.0",
        "grpcio-status>=1.51.0",
        "googleapis-common-protos>=1.58.0",
    ],

    "encryption": [
        # Additional encryption libraries
        "pycryptodome>=3.17.0",
        "bcrypt>=4.0.0",
        "argon2-cffi>=21.3.0",
    ],

    "monitoring": [
        # Enhanced monitoring and metrics
        "prometheus-client>=0.16.0",
        "statsd>=4.0.0",
        "opentelemetry-api>=1.16.0",
        "opentelemetry-sdk>=1.16.0",
    ],

    "cloud": [
        # Cloud service integrations
        "boto3>=1.26.0",  # AWS
        "google-cloud-core>=2.3.0",  # Google Cloud
        "azure-identity>=1.12.0",  # Azure
    ]
}

# All extras combined
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Python version requirement
PYTHON_REQUIRES = ">=3.8"

# Package classifiers
CLASSIFIERS = [
    # Development status
    "Development Status :: 4 - Beta",

    # Intended audience
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",

    # Topic classifications
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Scientific/Engineering",
    "Topic :: Database",
    "Topic :: Internet :: WWW/HTTP :: Dynamic Content",

    # License
    "License :: OSI Approved :: MIT License",

    # Programming language
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",

    # Operating systems
    "Operating System :: OS Independent",
    "Operating System :: POSIX",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",

    # Framework
    "Framework :: Jupyter",
    "Framework :: Jupyter :: JupyterLab",
    "Framework :: IPython",

    # Environment
    "Environment :: Web Environment",
    "Environment :: Console",

    # Natural language
    "Natural Language :: English",

    # Typing
    "Typing :: Typed",
]

# Keywords for PyPI
KEYWORDS = [
    "jupyter", "notebook", "sql", "database", "magic", "ipython",
    "data-science", "analytics", "query", "rest-api", "grpc",
    "security", "authentication", "validation", "async", "enterprise"
]

# Entry points for Jupyter extensions
ENTRY_POINTS = {
    "jupyter_serverextension": [
        "syne_sql_extension = syne_sql_extension.serverextension:_jupyter_server_extension_paths"
    ],
    "jupyter_nbextension": [
        "syne_sql_extension = syne_sql_extension.nbextension:_jupyter_nbextension_paths"
    ]
}

# Data files to include
PACKAGE_DATA = {
    "syne_sql_extension": [
        "static/**/*",
        "templates/**/*",
        "config/*.json",
        "config/*.yaml",
        "schemas/*.json",
        "py.typed",  # PEP 561 marker file
    ]
}

# Additional data files
DATA_FILES = [
    ("share/jupyter/nbextensions/syne_sql_extension", [
        "syne_sql_extension/static/extension.js",
        "syne_sql_extension/static/extension.css",
    ]),
    ("etc/jupyter/jupyter_notebook_config.d", [
        "syne_sql_extension/config/syne_sql_extension.json"
    ]),
]

setup(
    # Basic package information
    name="syne-sql-extension",
    version=get_version(),
    description="Enterprise-grade Jupyter extension for secure SQL query execution through internal services",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",

    # Author information
    author="SyneHQ",
    author_email="dev@synehq.com",
    maintainer="SyneHQ",
    maintainer_email="engineering@synehq.com",

    # URLs
    url="https://github.com/synehq/jupyter.sql-extension",
    project_urls={
        "Homepage": "https://github.com/synehq/jupyter.sql-extension",
        "Documentation": "https://jupyter-sql-extension.readthedocs.io/",
        "Repository": "https://github.com/synehq/jupyter.sql-extension",
        "Bug Reports": "https://github.com/synehq/jupyter.sql-extension/issues",
        "Changelog": "https://github.com/synehq/jupyter.sql-extension/blob/main/CHANGELOG.md",
        "Funding": "https://github.com/sponsors/synehq",
    },

    # License
    license="MIT",

    # Package discovery
    packages=find_packages(
        exclude=["tests*", "docs*", "examples*", "scripts*"]
    ),
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    include_package_data=True,
    zip_safe=False,  # Jupyter extensions need to be extracted

    # Dependencies
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    # Entry points
    entry_points=ENTRY_POINTS,

    # Metadata
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    platforms=["any"],

    # Configuration for modern tools
    cmdclass={},

    # Additional options for setuptools
    options={
        "bdist_wheel": {
            "universal": False,  # This is a Python 3 only package
        },
        "build_py": {
            "compile": True,
            "optimize": 1,
        },
        "egg_info": {
            "tag_build": None,
            "tag_date": None,
        },
    },

    # Jupyter-specific metadata
    jupyter_extension_metadata={
        "name": "syne_sql_extension",
        "description": "Secure SQL query execution through internal services",
        "version": get_version(),
        "requires": {
            "notebook": ">=6.4.0",
            "jupyter": ">=1.0.0",
        },
        "keywords": KEYWORDS,
        "homepage": "https://github.com/diagonal/jupyter-sql-extension",
    }
)
