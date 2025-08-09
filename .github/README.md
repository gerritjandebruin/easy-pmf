# GitHub Configuration

This directory contains GitHub-specific configuration files and templates for the Easy PMF project.

## 📁 Structure

```
.github/
├── ISSUE_TEMPLATE/          # Issue templates for different types of issues
│   ├── bug_report.yml       # Bug report template
│   ├── feature_request.yml  # Feature request template
│   ├── question.yml         # Question template
│   └── other.yml           # General issue template
├── workflows/              # GitHub Actions workflows
│   ├── ci.yml             # Main CI/CD pipeline
│   ├── publish.yml        # PyPI publishing workflow
│   ├── docs.yml           # Documentation deployment
│   ├── dependencies.yml   # Dependency management
│   ├── release.yml        # Release automation
│   └── housekeeping.yml   # Repository maintenance
├── instructions/          # AI assistant instructions
│   ├── environment.instructions.md  # Environment setup
│   ├── gh.instructions.md          # GitHub CLI usage
│   └── uv.instructions.md          # UV package manager
├── pull_request_template.md  # Pull request template
└── README.md                # This file
```

## 🚀 Workflows

### CI/CD Pipeline (`ci.yml`)
- **Triggers**: Push to main/develop, pull requests
- **Matrix Testing**: Python 3.9-3.12 on Ubuntu, Windows, macOS
- **Quality Checks**: Ruff linting/formatting, mypy type checking
- **Security**: Safety and bandit scans
- **Documentation**: MkDocs build verification
- **Package Build**: Distribution creation and validation

### Publishing (`publish.yml`)
- **Triggers**: GitHub releases, manual dispatch
- **Validation**: Version checks, full test suite
- **Publishing**: PyPI deployment with trusted publishing
- **Verification**: Post-publish installation testing
- **Assets**: Automatic release asset uploads

### Documentation (`docs.yml`)
- **Triggers**: Docs changes, releases, manual dispatch
- **Deployment**: GitHub Pages via MkDocs
- **Quality**: Link checking, markdown validation
- **Preview**: PR documentation previews

### Dependencies (`dependencies.yml`)
- **Schedule**: Weekly dependency updates
- **Security**: Vulnerability scanning with safety/bandit
- **Automation**: Automatic PR creation for updates
- **Analysis**: License and dependency tree reports

### Release Management (`release.yml`)
- **Manual**: Workflow dispatch with version input
- **Validation**: Version format and changelog checks
- **Automation**: Version bumping, tag creation, release publishing
- **Integration**: Triggers publishing workflow

### Housekeeping (`housekeeping.yml`)
- **Schedule**: Daily maintenance tasks
- **Stale Management**: Auto-mark and close inactive issues/PRs
- **Cleanup**: Old workflow run removal
- **Reporting**: Activity summaries

## 📋 Issue Templates

### Bug Report (`bug_report.yml`)
Structured template for bug reports including:
- Version information
- Reproduction steps
- Environment details
- Error messages

### Feature Request (`feature_request.yml`)
Template for feature suggestions with:
- Problem description
- Proposed solution
- Use case scenarios
- Priority assessment

### Question (`question.yml`)
Template for user questions including:
- Context and attempted solutions
- Code examples
- Version information

### Other (`other.yml`)
General template for issues not covered by specific templates.

## 🔧 Development Tools

### Pre-commit Configuration (`.pre-commit-config.yaml`)
Automated code quality checks including:
- **Ruff**: Linting and formatting
- **MyPy**: Type checking
- **Security**: Bandit scanning
- **Documentation**: Docstring coverage
- **General**: File formatting, trailing whitespace, etc.

Install with:
```bash
pip install pre-commit
pre-commit install
```

### Pull Request Template
Standardized PR template ensuring:
- Clear change description
- Testing confirmation
- Code quality checklist
- Breaking change documentation

## 🛡️ Security

### Automated Security Scanning
- **Safety**: Known vulnerability detection
- **Bandit**: Security issue identification
- **Dependency Analysis**: License and security monitoring

### Protected Branches
Main branch protection includes:
- Required PR reviews
- Status check requirements
- No direct pushes
- Linear history enforcement

## 📖 Instructions

AI assistant instructions for consistent development practices:
- **Environment**: PowerShell on Windows configuration
- **GitHub CLI**: Workflow and issue management
- **UV**: Modern Python package management

## 🎯 Best Practices

### For Contributors
1. **Use Issue Templates**: Select appropriate template for bug reports, features, or questions
2. **Follow PR Template**: Complete all sections of the PR template
3. **Enable Pre-commit**: Install pre-commit hooks for automatic quality checks
4. **Test Locally**: Run full test suite before submitting PRs

### For Maintainers
1. **Review Workflows**: Regularly review and update workflow configurations
2. **Monitor Security**: Check security scan results and address vulnerabilities
3. **Manage Dependencies**: Review automated dependency update PRs
4. **Release Process**: Use the release workflow for consistent versioning

## 🔄 Maintenance

### Regular Tasks
- **Weekly**: Review dependency update PRs
- **Monthly**: Check workflow performance and update versions
- **Quarterly**: Review and update templates and documentation

### Monitoring
- **CI/CD Health**: Monitor workflow success rates
- **Security Alerts**: Respond to vulnerability notifications
- **Community Engagement**: Manage issues and PR discussions

## 📚 Documentation

For more information:
- [Contributing Guidelines](../docs/contributing/guidelines.md)
- [Testing Guide](../docs/contributing/testing.md)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
