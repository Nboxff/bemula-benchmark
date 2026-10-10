# Bemula evaluation report: langmem

- Instances: 50  |  Items: 1742
- RA gate: 35/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.920 | 1.000 |
| FE (conditioned) | 0.406 | 0.479 |
| FE (raw) | 0.433 | 0.479 |
| PR (conditioned) | 0.740 | 0.790 |
| PR (raw) | 0.847 | 0.876 |
| RLA | 0.895 | 0.873 |
| AR (avg) | 0.357 | 0.412 |
| FA (forgetting ability) | 0.381 | 0.444 |
| UG (usability guard) | 0.814 | 0.831 |
| **Bemula Score** | 0.310 | 0.369 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.015

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.345 |
| indirect | 0.301 |
| jailbreak | 0.466 |
| multihop | 0.272 |
| roleplay | 0.403 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.700 |
| efficacy | 0.160 |
| preservation | 0.400 |
| relearning | 0.755 |
| attack | 0.020 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.872 | 0.436 | -- | 0.906 | 0.531 | -- |
| F2 | 0.940 | 0.320 | -- | 0.957 | 0.342 | -- |
| F3 | 0.962 | 0.354 | -- | 0.808 | 0.346 | -- |
| F4 | 0.692 | 0.308 | -- | 0.923 | 0.323 | -- |
| F5 | 0.909 | 0.318 | -- | 0.900 | 0.343 | -- |
| F6 | 0.952 | 0.905 | -- | -- | 0.200 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.906 | 0.292 | 0.708 | 0.939 | 0.322 | 0.250 |
| medium | 0.938 | 0.406 | 0.708 | 0.879 | 0.373 | 0.307 |
| short | 0.975 | 0.600 | 0.844 | 0.842 | 0.467 | 0.446 |
| xlong | 0.833 | 0.542 | 0.556 | 0.846 | 0.255 | 0.255 |
