# AGENTS.md

## Cursor Cloud specific instructions

This repository ("KIP") is currently an empty scaffold with only a `README.md`. There are no services, dependencies, build systems, or test frameworks configured yet.

### Current state
- **No source code** or application logic exists.
- **No package manager** lockfile or manifest (no `package.json`, `requirements.txt`, `go.mod`, etc.).
- **No Docker/container** configuration.
- **No CI/CD** pipeline configured.
- **No tests** or linting tools set up.

### When code is added
Once actual code and a dependency manifest are introduced, the update script (`SetupVmEnvironment`) and this section should be updated to reflect:
1. How to install dependencies (e.g., `npm install`, `pip install -r requirements.txt`).
2. How to run the dev server(s).
3. How to run lint and tests.
4. Any non-obvious environment variables or secrets required.
