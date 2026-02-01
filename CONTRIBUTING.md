# Contributing to ModelOptStreaming

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/yourusername/ModelOptStreaming.git
cd ModelOptStreaming
pip install -e ".[dev]"
```

## Code Style

We use:
- **Black** for formatting (100 char line length)
- **Ruff** for linting
- **Type hints** for all public APIs

Run before committing:
```bash
black modelopt_streaming/
ruff check modelopt_streaming/
```

## Testing

```bash
pytest tests/
```

## Adding New Quantization Formats

1. Create format implementation in `modelopt_streaming/<format>.py`
2. Add format to `QuantizationFormat` enum in `formats.py`
3. Update `get_quantization_config()` with format-specific config
4. Add tests in `tests/test_<format>.py`
5. Update README with format documentation

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with descriptive messages
6. Push to your fork
7. Open a Pull Request

## Questions?

Open an issue or discussion on GitHub.
