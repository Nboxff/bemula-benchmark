# Bemula evaluation report: lightmem

- Instances: 50  |  Items: 1742
- RA gate: 35/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.929 | 1.000 |
| FE (conditioned) | 0.179 | 0.244 |
| FE (raw) | 0.196 | 0.244 |
| PR (conditioned) | 0.880 | 0.895 |
| PR (raw) | 0.920 | 0.933 |
| RLA | 0.807 | 0.846 |
| AR (avg) | 0.180 | 0.205 |
| FA (forgetting ability) | 0.179 | 0.224 |
| UG (usability guard) | 0.843 | 0.870 |
| **Bemula Score** | 0.151 | 0.195 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.141 |
| indirect | 0.107 |
| jailbreak | 0.257 |
| multihop | 0.160 |
| roleplay | 0.233 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.700 |
| efficacy | 0.040 |
| preservation | 0.640 |
| relearning | 0.633 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.923 | 0.179 | -- | 0.781 | 0.287 | -- |
| F2 | 0.980 | 0.160 | -- | 0.826 | 0.196 | -- |
| F3 | 0.924 | 0.038 | -- | 0.846 | 0.119 | -- |
| F4 | 0.692 | 0.231 | -- | 0.923 | 0.231 | -- |
| F5 | 0.955 | 0.000 | -- | 0.700 | 0.143 | -- |
| F6 | 0.952 | 0.905 | -- | -- | 0.200 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.875 | 0.125 | 0.854 | 0.755 | 0.172 | 0.118 |
| medium | 0.969 | 0.203 | 0.917 | 0.788 | 0.180 | 0.163 |
| short | 1.000 | 0.250 | 0.889 | 1.000 | 0.205 | 0.214 |
| xlong | 0.917 | 0.208 | 0.778 | 0.769 | 0.164 | 0.143 |
