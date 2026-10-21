# Red-team report

| Defence | Attacks | Blocked | Allowed | False positives |
|---|---|---|---|---|
| Input guard | 9 | 9 | 0 | 0 of 100 benign |
| Tool escalation | 10 | 10 | 0 | n/a |
| Indirect injection (planted in retrieved text) | 44 | 44 | 0 | n/a |
| Unauthorized retrieval | 4 | 4 | 0 | n/a |

Indirect injection attack success: **0/44** (95% upper bound 8.0%; a bound under 1% would need roughly 380+ clean trials).
