# Bemula evaluation report: fullcontext

- Instances: 50  |  Items: 1742
- RA gate: 24/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.844 | 1.000 |
| FE (conditioned) | 0.268 | 0.333 |
| FE (raw) | 0.388 | 0.333 |
| PR (conditioned) | 0.733 | 0.736 |
| PR (raw) | 0.780 | 0.778 |
| RLA | 0.982 | 0.976 |
| AR (avg) | 0.276 | 0.297 |
| FA (forgetting ability) | 0.272 | 0.315 |
| UG (usability guard) | 0.849 | 0.847 |
| **Bemula Score** | 0.231 | 0.267 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.364 |
| indirect | 0.301 |
| jailbreak | 0.097 |
| multihop | 0.335 |
| roleplay | 0.282 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.480 |
| efficacy | 0.120 |
| preservation | 0.420 |
| relearning | 0.959 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.769 | 0.231 | -- | 1.000 | 0.287 | -- |
| F2 | 0.940 | 0.240 | -- | 0.957 | 0.284 | -- |
| F3 | 0.797 | 0.139 | -- | 1.000 | 0.300 | -- |
| F4 | 0.615 | 0.385 | -- | 1.000 | 0.169 | -- |
| F5 | 0.909 | 0.091 | -- | 0.950 | 0.333 | -- |
| F6 | 1.000 | 1.000 | -- | -- | 0.162 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.823 | 0.271 | 0.667 | 1.000 | 0.287 | 0.228 |
| medium | 0.906 | 0.266 | 0.708 | 0.939 | 0.283 | 0.224 |
| short | 0.875 | 0.250 | 0.800 | 1.000 | 0.287 | 0.240 |
| xlong | 0.708 | 0.292 | 0.889 | 1.000 | 0.191 | 0.222 |
