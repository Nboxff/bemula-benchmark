# Bemula evaluation report: lightmem

- Instances: 50  |  Items: 1742
- RA gate: 37/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.929 | 1.000 |
| FE (conditioned) | 0.089 | 0.074 |
| FE (raw) | 0.107 | 0.074 |
| PR (conditioned) | 0.860 | 0.883 |
| PR (raw) | 0.900 | 0.919 |
| RLA | 0.728 | 0.734 |
| AR (avg) | 0.056 | 0.062 |
| FA (forgetting ability) | 0.071 | 0.068 |
| UG (usability guard) | 0.791 | 0.805 |
| **Bemula Score** | 0.056 | 0.054 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.029 |
| indirect | 0.063 |
| jailbreak | 0.029 |
| multihop | 0.078 |
| roleplay | 0.083 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.740 |
| efficacy | 0.020 |
| preservation | 0.660 |
| relearning | 0.490 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.949 | 0.000 | -- | 0.719 | 0.087 | -- |
| F2 | 0.940 | 0.020 | -- | 0.739 | 0.080 | -- |
| F3 | 0.924 | 0.000 | -- | 0.731 | 0.049 | -- |
| F4 | 0.846 | 0.000 | -- | 0.846 | 0.077 | -- |
| F5 | 0.955 | 0.000 | -- | 0.650 | 0.019 | -- |
| F6 | 0.905 | 0.905 | -- | -- | 0.010 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.896 | 0.073 | 0.771 | 0.653 | 0.042 | 0.039 |
| medium | 0.969 | 0.094 | 0.917 | 0.697 | 0.057 | 0.058 |
| short | 0.950 | 0.100 | 0.889 | 0.947 | 0.087 | 0.086 |
| xlong | 0.917 | 0.125 | 0.889 | 0.769 | 0.055 | 0.068 |
