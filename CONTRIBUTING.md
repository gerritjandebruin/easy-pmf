# Contributing to Easy PMF

Thank you for your interest in contributing to Easy PMF! 🎉

## Quick Start

1. **Fork and clone** the repository
2. **Set up development environment**:
   ```bash
   # Install uv if needed: https://docs.astral.sh/uv/getting-started/installation/
   uv sync --all-extras
   uv run pre-commit install
   ```
3. **Make your changes** and test them:
   ```bash
   uv run pytest
   uv run pre-commit run --all-files
   ```
4. **Submit a pull request**

## Development Infrastructure

This project features a comprehensive CI/CD setup:

- ✅ **Automated Testing**: Matrix testing across Python 3.9-3.12 on multiple platforms
- ✅ **Code Quality**: Pre-commit hooks with ruff (linting/formatting) and mypy (type checking)
- ✅ **Security Scanning**: Automated vulnerability detection
- ✅ **Documentation**: Auto-deployment to [GitHub Pages](https://gerritjandebruin.github.io/easy-pmf/)
- ✅ **Publishing**: Automated PyPI releases on tags

## Detailed Guidelines

For comprehensive contributing information, including:

- Code style standards
- Testing requirements
- Documentation guidelines
- Pull request process
- Issue reporting

Please see our **[Detailed Contributing Guidelines](docs/contributing/guidelines.md)**.

## Types of Contributions

We welcome:

- 🐛 **Bug fixes** - Fix issues in existing code
- ✨ **New features** - Add functionality to the package
- 📝 **Documentation** - Improve guides, examples, or API docs
- 🧪 **Tests** - Improve test coverage
- 🎯 **Examples** - Add real-world analysis examples

## Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/gerritjandebruin/easy-pmf/discussions)
- **Bugs**: Report via [GitHub Issues](https://github.com/gerritjandebruin/easy-pmf/issues)
- **Documentation**: Visit [our docs](https://gerritjandebruin.github.io/easy-pmf/)

## Code of Conduct

Please be respectful and constructive in all interactions. We strive to maintain a welcoming environment for all contributors.

---

**Easy PMF** - Making positive matrix factorization accessible to everyone! 🌍
