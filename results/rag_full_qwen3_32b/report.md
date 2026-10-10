# Bemula evaluation report: rag

- Instances: 50  |  Items: 1742
- RA gate: 16/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.732 | 1.000 |
| FE (conditioned) | 0.071 | 0.064 |
| FE (raw) | 0.192 | 0.064 |
| PR (conditioned) | 0.633 | 0.792 |
| PR (raw) | 0.640 | 0.812 |
| RLA | 0.728 | 1.000 |
| AR (avg) | 0.050 | 0.070 |
| FA (forgetting ability) | 0.059 | 0.067 |
| UG (usability guard) | 0.679 | 0.890 |
| **Bemula Score** | 0.040 | 0.059 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.024 |
| indirect | 0.063 |
| jailbreak | 0.087 |
| multihop | 0.034 |
| roleplay | 0.039 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.320 |
| efficacy | 0.020 |
| preservation | 0.220 |
| relearning | 0.571 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.744 | 0.026 | -- | 0.719 | 0.056 | -- |
| F2 | 0.760 | 0.000 | -- | 0.826 | 0.098 | -- |
| F3 | 0.734 | 0.013 | -- | 0.731 | 0.022 | -- |
| F4 | 0.231 | 0.000 | -- | 0.692 | 0.015 | -- |
| F5 | 0.818 | 0.045 | -- | 0.650 | 0.048 | -- |
| F6 | 0.857 | 0.619 | -- | -- | 0.057 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.677 | 0.042 | 0.438 | 0.612 | 0.056 | 0.025 |
| medium | 0.766 | 0.078 | 0.750 | 0.848 | 0.043 | 0.046 |
| short | 0.875 | 0.125 | 0.733 | 1.000 | 0.067 | 0.078 |
| xlong | 0.625 | 0.083 | 0.556 | 0.462 | 0.009 | 0.014 |
