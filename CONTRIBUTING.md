# Contributing to NeuroAdapt-X

Thank you for your interest in contributing to NeuroAdapt-X!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/josephjilovec/neuroadapt-x.git
cd neuroadapt-x
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install in development mode:
```bash
pip install -e .
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Maximum line length: 120 characters

## Testing

Run tests with pytest:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Update documentation if needed
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Code Review

All submissions require review. We use GitHub pull requests for this purpose.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

