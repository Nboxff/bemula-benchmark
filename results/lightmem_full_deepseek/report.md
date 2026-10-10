# Bemula evaluation report: lightmem

- Instances: 50  |  Items: 1742
- RA gate: 39/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.946 | 1.000 |
| FE (conditioned) | 0.129 | 0.124 |
| FE (raw) | 0.152 | 0.124 |
| PR (conditioned) | 0.940 | 0.949 |
| PR (raw) | 0.940 | 0.949 |
| RLA | 0.825 | 0.829 |
| AR (avg) | 0.104 | 0.105 |
| FA (forgetting ability) | 0.116 | 0.114 |
| UG (usability guard) | 0.880 | 0.887 |
| **Bemula Score** | 0.102 | 0.101 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.029 |
| indirect | 0.097 |
| jailbreak | 0.175 |
| multihop | 0.097 |
| roleplay | 0.121 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.780 |
| efficacy | 0.080 |
| preservation | 0.820 |
| relearning | 0.633 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.949 | 0.103 | -- | 0.781 | 0.144 | -- |
| F2 | 0.960 | 0.040 | -- | 0.870 | 0.084 | -- |
| F3 | 0.949 | 0.000 | -- | 0.885 | 0.111 | -- |
| F4 | 0.769 | 0.231 | -- | 0.923 | 0.138 | -- |
| F5 | 0.955 | 0.000 | -- | 0.700 | 0.076 | -- |
| F6 | 1.000 | 0.952 | -- | -- | 0.067 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.938 | 0.104 | 0.917 | 0.816 | 0.096 | 0.087 |
| medium | 0.953 | 0.141 | 0.979 | 0.727 | 0.103 | 0.102 |
| short | 0.975 | 0.175 | 0.956 | 0.947 | 0.118 | 0.137 |
| xlong | 0.917 | 0.125 | 0.778 | 0.923 | 0.109 | 0.099 |
