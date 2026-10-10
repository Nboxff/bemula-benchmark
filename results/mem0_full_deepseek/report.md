# Bemula evaluation report: mem0

- Instances: 50  |  Items: 1742
- RA gate: 31/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.879 | 1.000 |
| FE (conditioned) | 0.170 | 0.172 |
| FE (raw) | 0.183 | 0.172 |
| PR (conditioned) | 0.867 | 0.903 |
| PR (raw) | 0.900 | 0.925 |
| RLA | 0.711 | 0.770 |
| AR (avg) | 0.126 | 0.127 |
| FA (forgetting ability) | 0.146 | 0.148 |
| UG (usability guard) | 0.785 | 0.834 |
| **Bemula Score** | 0.115 | 0.124 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.087 |
| indirect | 0.058 |
| jailbreak | 0.189 |
| multihop | 0.136 |
| roleplay | 0.160 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.620 |
| efficacy | 0.060 |
| preservation | 0.640 |
| relearning | 0.490 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.897 | 0.077 | -- | 0.750 | 0.156 | -- |
| F2 | 0.900 | 0.160 | -- | 0.783 | 0.178 | -- |
| F3 | 0.886 | 0.000 | -- | 0.654 | 0.068 | -- |
| F4 | 0.692 | 0.385 | -- | 0.846 | 0.231 | -- |
| F5 | 0.909 | 0.091 | -- | 0.550 | 0.171 | -- |
| F6 | 0.857 | 0.952 | -- | -- | 0.067 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.865 | 0.146 | 0.833 | 0.633 | 0.120 | 0.096 |
| medium | 0.875 | 0.188 | 0.896 | 0.727 | 0.147 | 0.134 |
| short | 0.950 | 0.200 | 0.889 | 0.947 | 0.128 | 0.147 |
| xlong | 0.833 | 0.167 | 0.778 | 0.615 | 0.091 | 0.085 |
