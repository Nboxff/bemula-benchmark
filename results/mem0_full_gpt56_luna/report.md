# Bemula evaluation report: mem0

- Instances: 50  |  Items: 1742
- RA gate: 36/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.897 | 1.000 |
| FE (conditioned) | 0.085 | 0.097 |
| FE (raw) | 0.107 | 0.097 |
| PR (conditioned) | 0.800 | 0.824 |
| PR (raw) | 0.840 | 0.852 |
| RLA | 0.772 | 0.789 |
| AR (avg) | 0.047 | 0.045 |
| FA (forgetting ability) | 0.063 | 0.066 |
| UG (usability guard) | 0.786 | 0.806 |
| **Bemula Score** | 0.049 | 0.053 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.044 |
| indirect | 0.024 |
| jailbreak | 0.078 |
| multihop | 0.049 |
| roleplay | 0.039 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.720 |
| efficacy | 0.020 |
| preservation | 0.520 |
| relearning | 0.571 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.872 | 0.000 | -- | 0.719 | 0.037 | -- |
| F2 | 0.860 | 0.000 | -- | 0.826 | 0.036 | -- |
| F3 | 0.911 | 0.013 | -- | 0.731 | 0.046 | -- |
| F4 | 0.692 | 0.000 | -- | 0.923 | 0.046 | -- |
| F5 | 1.000 | 0.000 | -- | 0.750 | 0.086 | -- |
| F6 | 1.000 | 0.857 | -- | -- | 0.048 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.865 | 0.062 | 0.750 | 0.735 | 0.049 | 0.041 |
| medium | 0.953 | 0.109 | 0.833 | 0.758 | 0.033 | 0.048 |
| short | 0.975 | 0.100 | 0.867 | 1.000 | 0.056 | 0.070 |
| xlong | 0.750 | 0.083 | 0.556 | 0.615 | 0.055 | 0.039 |
