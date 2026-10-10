# Bemula evaluation report: simplemem

- Instances: 50  |  Items: 1742
- RA gate: 34/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.906 | 1.000 |
| FE (conditioned) | 0.179 | 0.234 |
| FE (raw) | 0.237 | 0.234 |
| PR (conditioned) | 0.847 | 0.833 |
| PR (raw) | 0.873 | 0.853 |
| RLA | 0.904 | 0.884 |
| AR (avg) | 0.200 | 0.226 |
| FA (forgetting ability) | 0.189 | 0.230 |
| UG (usability guard) | 0.875 | 0.858 |
| **Bemula Score** | 0.165 | 0.197 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.007

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.121 |
| indirect | 0.180 |
| jailbreak | 0.291 |
| multihop | 0.160 |
| roleplay | 0.248 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.680 |
| efficacy | 0.080 |
| preservation | 0.620 |
| relearning | 0.796 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.846 | 0.128 | -- | 0.812 | 0.325 | -- |
| F2 | 0.940 | 0.200 | -- | 0.957 | 0.316 | -- |
| F3 | 0.937 | 0.025 | -- | 0.962 | 0.135 | -- |
| F4 | 0.538 | 0.308 | -- | 1.000 | 0.200 | -- |
| F5 | 0.955 | 0.045 | -- | 0.850 | 0.105 | -- |
| F6 | 1.000 | 0.857 | -- | -- | 0.086 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.885 | 0.125 | 0.750 | 0.857 | 0.176 | 0.119 |
| medium | 0.938 | 0.219 | 0.875 | 0.939 | 0.223 | 0.200 |
| short | 0.925 | 0.250 | 0.933 | 0.947 | 0.226 | 0.223 |
| xlong | 0.875 | 0.167 | 0.778 | 0.923 | 0.182 | 0.147 |
