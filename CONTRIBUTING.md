# Contributing

Contributions should include a focused issue, tests for behavioral changes, and raw evidence for performance claims. Run:

```bash
ruff check src tests experiments
pytest
python -m kvbridge demo --min-r2 0.99
python -m build
```

Do not report synthetic reconstruction as real-model quality. New model adapters must pin a Transformers version, document cache layout and RoPE behavior, and include an end-to-end identity/control test before they are described as supported.
