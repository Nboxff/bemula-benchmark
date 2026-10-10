# Bemula evaluation report: langmem

- Instances: 50  |  Items: 1742
- RA gate: 0/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.219 | -- |
| FE (conditioned) | 0.147 | -- |
| FE (raw) | 0.728 | -- |
| PR (conditioned) | 0.053 | -- |
| PR (raw) | 0.100 | -- |
| RLA | 0.939 | -- |
| AR (avg) | 0.094 | -- |
| FA (forgetting ability) | 0.118 | -- |
| UG (usability guard) | 0.224 | -- |
| **Bemula Score** | 0.026 | -- |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.078 |
| indirect | 0.117 |
| jailbreak | 0.053 |
| multihop | 0.083 |
| roleplay | 0.141 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.000 |
| efficacy | 0.000 |
| preservation | 0.000 |
| relearning | 0.857 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.282 | 0.154 | -- | 0.938 | 0.119 | -- |
| F2 | 0.140 | 0.040 | -- | 0.913 | 0.067 | -- |
| F3 | 0.304 | 0.241 | -- | 1.000 | 0.127 | -- |
| F4 | 0.308 | 0.077 | -- | 1.000 | 0.123 | -- |
| F5 | 0.045 | 0.045 | -- | 0.850 | 0.038 | -- |
| F6 | 0.095 | 0.190 | -- | -- | 0.038 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.135 | 0.062 | 0.000 | 0.918 | 0.028 | 0.000 |
| medium | 0.344 | 0.250 | 0.062 | 0.970 | 0.177 | 0.052 |
| short | 0.325 | 0.250 | 0.111 | 1.000 | 0.164 | 0.068 |
| xlong | 0.042 | 0.042 | 0.000 | 0.846 | 0.000 | 0.000 |
