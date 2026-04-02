<!--
SPDX-License-Identifier: MIT
Copyright (c) 2026 Rohan R
Author: Rohan R
-->

# Contributing

Thanks for helping with NotionHealth AI.

## Before you start

1. Open an issue or start a discussion if the change is large
2. Keep changes focused
3. Avoid unrelated cleanup in the same PR

## Local setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

If you are working on TRIBEv2 support on Apple Silicon, use the arm64 environment described in [README.md](README.md).

## Useful commands

```bash
make serve-api
make test
make lint
```

## Pull requests

- Describe what changed and why
- Mention any tradeoffs
- Include screenshots for UI changes
- Call out follow-up work if you are leaving any behind

## Style

- Keep code changes small when possible
- Write docs in plain English
- Prefer direct wording over marketing copy
- Update tests when behavior changes

## Security

Do not commit secrets, tokens, or real `.env` values. If you find a security issue, use the process in [SECURITY.md](SECURITY.md).
