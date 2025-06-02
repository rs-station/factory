# Factory

A Python package for crystallographic data processing and analysis.

## Installation

```bash
pip install factory
```

## Features

- Crystallographic data processing
- Neural network-based profile modeling
- Background estimation
- Structure factor prediction

## Quick Start

```python
from factory import Model
from factory import distributions

# Create and use your model
model = Model()
```

## Development

To set up the development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/factory.git
cd factory

# Install development dependencies
pip install -e ".[test,docs]"

# Run tests
pytest

# Build documentation
cd docs
make html
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
