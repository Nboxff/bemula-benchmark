# Bemula evaluation report: amem

- Instances: 31  |  Items: 848
- RA gate: 20/31 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.827 | 1.000 |
| FE (conditioned) | 0.125 | 0.127 |
| FE (raw) | 0.183 | 0.127 |
| PR (conditioned) | 0.688 | 0.667 |
| PR (raw) | 0.699 | 0.683 |
| RLA | 0.846 | 0.871 |
| AR (avg) | 0.107 | 0.115 |
| FA (forgetting ability) | 0.116 | 0.121 |
| UG (usability guard) | 0.763 | 0.762 |
| **Bemula Score** | 0.088 | 0.092 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.040 |
| indirect | 0.162 |
| jailbreak | 0.081 |
| multihop | 0.131 |
| roleplay | 0.121 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.645 |
| efficacy | 0.032 |
| preservation | 0.355 |
| relearning | 0.733 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.750 | 0.083 | -- | 0.900 | 0.120 | -- |
| F2 | 0.926 | 0.000 | -- | 0.846 | 0.154 | -- |
| F3 | 0.921 | 0.053 | -- | 0.833 | 0.089 | -- |
| F4 | 0.143 | 0.000 | -- | 1.000 | 0.086 | -- |
| F5 | 0.900 | 0.000 | -- | 0.700 | 0.160 | -- |
| F6 | 0.700 | 1.000 | -- | -- | 0.000 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| medium | 0.766 | 0.141 | 0.688 | 0.788 | 0.090 | 0.083 |
| short | 0.925 | 0.100 | 0.689 | 0.947 | 0.133 | 0.093 |
