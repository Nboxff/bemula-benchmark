# Bemula evaluation report: simplemem

- Instances: 50  |  Items: 1742
- RA gate: 19/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.786 | 1.000 |
| FE (conditioned) | 0.143 | 0.234 |
| FE (raw) | 0.241 | 0.234 |
| PR (conditioned) | 0.547 | 0.561 |
| PR (raw) | 0.633 | 0.702 |
| RLA | 0.833 | 0.900 |
| AR (avg) | 0.140 | 0.151 |
| FA (forgetting ability) | 0.141 | 0.188 |
| UG (usability guard) | 0.675 | 0.711 |
| **Bemula Score** | 0.095 | 0.134 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.049 |
| indirect | 0.214 |
| jailbreak | 0.068 |
| multihop | 0.189 |
| roleplay | 0.180 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.380 |
| efficacy | 0.040 |
| preservation | 0.160 |
| relearning | 0.673 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.590 | 0.128 | -- | 0.844 | 0.138 | -- |
| F2 | 0.880 | 0.080 | -- | 0.870 | 0.209 | -- |
| F3 | 0.823 | 0.025 | -- | 0.846 | 0.114 | -- |
| F4 | 0.615 | 0.077 | -- | 0.923 | 0.215 | -- |
| F5 | 0.818 | 0.091 | -- | 0.700 | 0.152 | -- |
| F6 | 0.857 | 0.857 | -- | -- | 0.029 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.792 | 0.135 | 0.458 | 0.816 | 0.125 | 0.079 |
| medium | 0.797 | 0.141 | 0.562 | 0.848 | 0.133 | 0.095 |
| short | 0.825 | 0.200 | 0.556 | 1.000 | 0.205 | 0.151 |
| xlong | 0.667 | 0.083 | 0.889 | 0.615 | 0.100 | 0.068 |
