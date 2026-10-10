# Bemula evaluation report: rag

- Instances: 50  |  Items: 1742
- RA gate: 15/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.728 | 1.000 |
| FE (conditioned) | 0.125 | 0.156 |
| FE (raw) | 0.290 | 0.156 |
| PR (conditioned) | 0.647 | 0.800 |
| PR (raw) | 0.660 | 0.822 |
| RLA | 0.711 | 0.870 |
| AR (avg) | 0.118 | 0.138 |
| FA (forgetting ability) | 0.122 | 0.146 |
| UG (usability guard) | 0.678 | 0.834 |
| **Bemula Score** | 0.082 | 0.122 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.087 |
| indirect | 0.083 |
| jailbreak | 0.199 |
| multihop | 0.078 |
| roleplay | 0.146 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.300 |
| efficacy | 0.020 |
| preservation | 0.260 |
| relearning | 0.510 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.718 | 0.051 | -- | 0.719 | 0.144 | -- |
| F2 | 0.780 | 0.080 | -- | 0.870 | 0.169 | -- |
| F3 | 0.734 | 0.000 | -- | 0.615 | 0.100 | -- |
| F4 | 0.154 | 0.077 | -- | 0.692 | 0.092 | -- |
| F5 | 0.773 | 0.045 | -- | 0.650 | 0.076 | -- |
| F6 | 0.905 | 0.952 | -- | -- | 0.095 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.656 | 0.115 | 0.458 | 0.633 | 0.111 | 0.061 |
| medium | 0.766 | 0.125 | 0.729 | 0.818 | 0.120 | 0.095 |
| short | 0.850 | 0.175 | 0.778 | 0.895 | 0.133 | 0.127 |
| xlong | 0.708 | 0.083 | 0.556 | 0.462 | 0.118 | 0.050 |
