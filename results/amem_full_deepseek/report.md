# Bemula evaluation report: amem

- Instances: 50  |  Items: 1742
- RA gate: 20/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.732 | 1.000 |
| FE (conditioned) | 0.295 | 0.359 |
| FE (raw) | 0.433 | 0.359 |
| PR (conditioned) | 0.540 | 0.667 |
| PR (raw) | 0.633 | 0.750 |
| RLA | 0.930 | 0.938 |
| AR (avg) | 0.212 | 0.207 |
| FA (forgetting ability) | 0.250 | 0.273 |
| UG (usability guard) | 0.709 | 0.791 |
| **Bemula Score** | 0.177 | 0.215 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.223 |
| indirect | 0.175 |
| jailbreak | 0.257 |
| multihop | 0.204 |
| roleplay | 0.199 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.400 |
| efficacy | 0.060 |
| preservation | 0.160 |
| relearning | 0.837 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.667 | 0.359 | -- | 0.875 | 0.338 | -- |
| F2 | 0.780 | 0.300 | -- | 0.957 | 0.364 | -- |
| F3 | 0.785 | 0.101 | -- | 0.923 | 0.127 | -- |
| F4 | 0.308 | 0.308 | -- | 1.000 | 0.108 | -- |
| F5 | 0.818 | 0.182 | -- | 0.950 | 0.171 | -- |
| F6 | 0.714 | 1.000 | -- | -- | 0.095 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.677 | 0.292 | 0.375 | 0.898 | 0.242 | 0.154 |
| medium | 0.719 | 0.281 | 0.583 | 0.909 | 0.200 | 0.173 |
| short | 0.950 | 0.350 | 0.711 | 1.000 | 0.205 | 0.226 |
| xlong | 0.625 | 0.250 | 0.333 | 1.000 | 0.136 | 0.107 |
