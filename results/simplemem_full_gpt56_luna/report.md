# Bemula evaluation report: simplemem

- Instances: 50  |  Items: 1742
- RA gate: 38/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.933 | 1.000 |
| FE (conditioned) | 0.161 | 0.158 |
| FE (raw) | 0.174 | 0.158 |
| PR (conditioned) | 0.827 | 0.842 |
| PR (raw) | 0.900 | 0.921 |
| RLA | 0.904 | 0.933 |
| AR (avg) | 0.187 | 0.232 |
| FA (forgetting ability) | 0.174 | 0.192 |
| UG (usability guard) | 0.864 | 0.887 |
| **Bemula Score** | 0.150 | 0.170 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.126 |
| indirect | 0.141 |
| jailbreak | 0.243 |
| multihop | 0.165 |
| roleplay | 0.262 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.760 |
| efficacy | 0.040 |
| preservation | 0.560 |
| relearning | 0.796 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.897 | 0.128 | -- | 0.875 | 0.294 | -- |
| F2 | 0.940 | 0.100 | -- | 0.913 | 0.227 | -- |
| F3 | 0.924 | 0.038 | -- | 0.962 | 0.081 | -- |
| F4 | 1.000 | 0.462 | -- | 1.000 | 0.446 | -- |
| F5 | 0.955 | 0.045 | -- | 0.800 | 0.219 | -- |
| F6 | 0.952 | 0.762 | -- | -- | 0.124 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.927 | 0.146 | 0.792 | 0.878 | 0.158 | 0.126 |
| medium | 0.953 | 0.172 | 0.812 | 0.909 | 0.220 | 0.167 |
| short | 0.975 | 0.175 | 0.889 | 1.000 | 0.226 | 0.187 |
| xlong | 0.833 | 0.167 | 0.778 | 0.846 | 0.145 | 0.126 |
