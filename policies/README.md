# Supporting Policy Documents

Drop your own policy documents (PDFs) into this folder. When a chat question or
answer references one of these topics, the matching document(s) are automatically
attached in the **Chat** tab as downloadable files.

A topic can map to **more than one file** (for example, SOC 2 attaches both the
executive summary and the technical report).

The app looks for the following filenames (see `detect_relevant_documents` in
`app.py` to customize the mapping, add topics, or change filenames):

| Topic / keywords | Expected filename(s) |
| --- | --- |
| privacy policy | `privacy-policy.pdf` |
| terms of service, tos | `terms-of-service.pdf` |
| consent to use of data | `consent-to-use-of-data.pdf` |
| insider trading, MNPI, code of ethics, personal trading | `insider-trading-policy.pdf` |
| information security, infosec | `information-security-policy.pdf` |
| incident response, breach | `incident-response-policy.pdf` |
| soc 2 | `soc2-executive-summary.pdf` **and** `soc2-technical-report.pdf` |
| data privacy, third party | `data-privacy-third-party-use-policy.pdf` |
| de-identification | `de-identification-guide.pdf` |
| hipaa, phi | `hipaa-compliance.pdf` |
| business associate agreement, baa | `business-associate-agreement.pdf` |
| cybersecurity | `cybersecurity-policy.pdf` |
| data security, encryption | `data-security-overview.pdf` |

Notes:
- These files are intentionally **not** committed to the repository. The
  `.gitignore` excludes `policies/*.pdf` so you never accidentally publish
  confidential documents.
- Matching is on filename, so the name on disk must match the mapping exactly.
  (An earlier version of the original app silently attached nothing because the
  files on disk had year suffixes like `... (2025).pdf` that no longer matched
  the mapping; the lookup now uses the canonical names above.)
- If a file is missing, the app simply skips attaching it. Nothing breaks.
- You can change the directory with the `POLICIES_DIR` environment variable.
