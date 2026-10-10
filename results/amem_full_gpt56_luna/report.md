# Bemula evaluation report: amem

- Instances: 50  |  Items: 1742
- RA gate: 21/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.804 | 1.000 |
| FE (conditioned) | 0.330 | 0.417 |
| FE (raw) | 0.415 | 0.417 |
| PR (conditioned) | 0.593 | 0.730 |
| PR (raw) | 0.660 | 0.810 |
| RLA | 0.886 | 0.838 |
| AR (avg) | 0.247 | 0.291 |
| FA (forgetting ability) | 0.285 | 0.348 |
| UG (usability guard) | 0.725 | 0.782 |
| **Bemula Score** | 0.207 | 0.272 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.165 |
| indirect | 0.272 |
| jailbreak | 0.248 |
| multihop | 0.272 |
| roleplay | 0.277 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.420 |
| efficacy | 0.060 |
| preservation | 0.240 |
| relearning | 0.735 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.821 | 0.462 | -- | 0.844 | 0.425 | -- |
| F2 | 0.840 | 0.520 | -- | 0.913 | 0.400 | -- |
| F3 | 0.797 | 0.051 | -- | 0.885 | 0.149 | -- |
| F4 | 0.462 | 0.308 | -- | 1.000 | 0.123 | -- |
| F5 | 0.864 | 0.045 | -- | 0.850 | 0.124 | -- |
| F6 | 0.857 | 1.000 | -- | -- | 0.190 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.740 | 0.302 | 0.354 | 0.857 | 0.214 | 0.140 |
| medium | 0.844 | 0.328 | 0.729 | 0.879 | 0.263 | 0.235 |
| short | 0.900 | 0.400 | 0.756 | 0.947 | 0.328 | 0.307 |
| xlong | 0.792 | 0.333 | 0.333 | 0.923 | 0.182 | 0.137 |
