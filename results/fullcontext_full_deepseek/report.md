# Bemula evaluation report: fullcontext

- Instances: 50  |  Items: 1742
- RA gate: 19/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.750 | 1.000 |
| FE (conditioned) | 0.536 | 0.714 |
| FE (raw) | 0.723 | 0.714 |
| PR (conditioned) | 0.460 | 0.632 |
| PR (raw) | 0.473 | 0.667 |
| RLA | 1.000 | 1.000 |
| AR (avg) | 0.329 | 0.400 |
| FA (forgetting ability) | 0.420 | 0.535 |
| UG (usability guard) | 0.678 | 0.795 |
| **Bemula Score** | 0.285 | 0.425 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.007

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.369 |
| indirect | 0.340 |
| jailbreak | 0.252 |
| multihop | 0.374 |
| roleplay | 0.311 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.380 |
| efficacy | 0.180 |
| preservation | 0.160 |
| relearning | 1.000 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.667 | 0.538 | -- | 1.000 | 0.362 | -- |
| F2 | 0.820 | 0.680 | -- | 1.000 | 0.364 | -- |
| F3 | 0.696 | 0.392 | -- | 1.000 | 0.308 | -- |
| F4 | 0.692 | 0.615 | -- | 1.000 | 0.154 | -- |
| F5 | 0.909 | 0.500 | -- | 1.000 | 0.410 | -- |
| F6 | 0.810 | 0.714 | -- | -- | 0.305 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.740 | 0.510 | 0.312 | 1.000 | 0.339 | 0.232 |
| medium | 0.766 | 0.594 | 0.438 | 1.000 | 0.290 | 0.274 |
| short | 0.925 | 0.575 | 0.711 | 1.000 | 0.385 | 0.397 |
| xlong | 0.458 | 0.417 | 0.111 | 1.000 | 0.300 | 0.118 |
