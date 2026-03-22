# Contributing to DQN Atari

Thank you for your interest in contributing to the DQN Atari project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions with other contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/Mugisha-isaac/Formative3-Group12-DQN.git
   cd Formative3-Group12-DQN
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies with dev tools**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

## Making Changes

### Code Style

We use:
- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **Flake8** for linting

Format your code before submitting:
```bash
black .
isort .
flake8 src/
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private**: Prefix with `_`

### Testing

Write tests for new functionality:
```bash
pytest tests/
pytest --cov=src tests/  # With coverage report
```

Tests should:
- Be in `tests/` directory
- Follow naming convention: `test_*.py`
- Use descriptive test names: `test_function_does_something()`

### Documentation

- Add docstrings to all public functions/classes:
  ```python
  def train_agent(env, timesteps=100000):
      """
      Train a DQN agent in the given environment.
      
      Args:
          env: The Gymnasium environment
          timesteps: Total training timesteps (default: 100000)
          
      Returns:
          The trained agent model
          
      Raises:
          ValueError: If timesteps is negative
      """
  ```

- Update README.md if adding features
- Add comments for complex logic

## Committing Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add hyperparameter tuning feature"
   ```

   Use conventional commit types:
   - `feat`: New feature
   - `fix`: Bug fix
   - `docs`: Documentation change
   - `refactor`: Code refactoring
   - `test`: Adding/updating tests
   - `perf`: Performance improvement
   - `chore`: Maintenance

3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Submitting a Pull Request

1. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Description of what changed and why
   - Reference to any related issues (e.g., "Closes #42")

2. **Ensure CI passes**:
   - All tests pass
   - Code is properly formatted
   - No linting errors

3. **Address feedback** from reviewers

4. **Merge** once approved

## Experiment and Hyperparameter Contributions

If contributing experimental results:

1. Document your hyperparameters in a table format
2. Include at least 3 runs for statistical validity
3. Report mean and standard deviation of results
4. Add screenshots or plots if available
5. Document any environment setup differences

Example format:
```
| Experiment | lr | gamma | batch_size | Mean Reward ± std | Notes |
| --- | --- | --- | --- | --- | --- |
| exp_new_1 | 0.0001 | 0.99 | 64 | 1.50 ± 0.30 | Baseline |
```

## Questions or Need Help?

- Open an issue for bug reports or feature requests
- Discuss major changes in issues before starting work
- Comment on PRs with questions

Thank you for contributing!
