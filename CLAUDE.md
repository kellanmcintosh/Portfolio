# Portfolio — Claude Code Guide

A collection of data science and ML projects by Kellan McIntosh.

## Projects

- **Credit Risk Scorer** — credit risk modeling project
- **mri_model** — MRI tumor detection model

## Git & Version Control

### Commit conventions
Use the imperative mood in the subject line, keep it under 72 characters, and optionally add a body for context:

```
Add logistic regression baseline to Credit Risk Scorer

Body explains *why*, not what — the diff already shows what changed.
```

Common prefixes:
- `Add` — new file, feature, or project
- `Update` — modify existing functionality
- `Fix` — bug fix
- `Remove` — delete code or files
- `Refactor` — restructure without changing behavior
- `Docs` — documentation only

### Branching
- `main` — stable, reviewed work only
- Feature branches: `feature/<short-description>`
- Fix branches: `fix/<short-description>`

### When to commit
- Commit when the user explicitly asks, or suggest a commit when a meaningful chunk of work is complete (new feature, bug fix, significant refactor) — not after every small edit
- Always stage specific files by name rather than `git add -A`
- Never commit `.DS_Store`, credentials, or `.env` files
- Never commit large model weights (`.h5`, `.pkl`, `.pt`, etc.) — document download instructions instead

## Working in this repo

- Python-based projects; each project lives in its own subdirectory
- Prefer editing existing files over creating new ones
- Do not create documentation files unless explicitly asked
