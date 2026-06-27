# Security Policy

## Scope

Manas is an experimental local-first research project. It runs entirely on
your own machine. It does not connect to any cloud service, does not require
an API key, and does not transmit any data externally by default.

Security issues most likely to apply:

- Malformed `.manas` files causing unexpected behavior on load
- Path traversal or unsafe file handling during folder ingestion
- Integer overflow or panic in binary format parsing
- Denial of service via crafted input causing unbounded memory growth

## Supported Versions

Only the latest version on the `main` branch is supported.

| Version | Supported |
|---|---|
| main | ✅ |
| older tags | ❌ |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues privately by emailing:

```
security@aarambhdevhub.com
```

Or via GitHub's private vulnerability reporting:
**Security → Report a vulnerability** on this repository.

Please include:

- A clear description of the vulnerability
- Steps to reproduce it
- The version or commit SHA affected
- Any proof-of-concept code if available

You will receive a response within 7 days.

## What to Expect

- Acknowledgment within 7 days
- Fix or mitigation within 30 days for confirmed issues
- Credit in the release notes if you wish

## Out of Scope

- Issues that require physical access to the machine
- Issues in future planned features not yet implemented
- Social engineering attacks
