# Bemula evaluation report: langmem

- Instances: 50  |  Items: 1742
- RA gate: 11/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.504 | 1.000 |
| FE (conditioned) | 0.379 | 0.765 |
| FE (raw) | 0.705 | 0.765 |
| PR (conditioned) | 0.200 | 0.455 |
| PR (raw) | 0.300 | 0.485 |
| RLA | 0.789 | 0.667 |
| AR (avg) | 0.256 | 0.600 |
| FA (forgetting ability) | 0.312 | 0.677 |
| UG (usability guard) | 0.397 | 0.550 |
| **Bemula Score** | 0.124 | 0.373 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.252 |
| indirect | 0.306 |
| jailbreak | 0.204 |
| multihop | 0.228 |
| roleplay | 0.291 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.220 |
| efficacy | 0.140 |
| preservation | 0.040 |
| relearning | 0.653 |
| attack | 0.020 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.538 | 0.385 | -- | 0.875 | 0.269 | -- |
| F2 | 0.440 | 0.340 | -- | 0.783 | 0.249 | -- |
| F3 | 0.608 | 0.430 | -- | 0.692 | 0.338 | -- |
| F4 | 0.385 | 0.231 | -- | 0.846 | 0.185 | -- |
| F5 | 0.364 | 0.136 | -- | 0.750 | 0.190 | -- |
| F6 | 0.429 | 0.619 | -- | -- | 0.076 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.396 | 0.302 | 0.021 | 0.837 | 0.195 | 0.032 |
| medium | 0.625 | 0.375 | 0.188 | 0.818 | 0.303 | 0.132 |
| short | 0.725 | 0.650 | 0.422 | 0.737 | 0.431 | 0.295 |
| xlong | 0.250 | 0.250 | 0.111 | 0.615 | 0.055 | 0.031 |
