# Bemula evaluation report: mem0

- Instances: 50  |  Items: 1742
- RA gate: 28/50 instances passed (all retention items correct)

| Metric | All instances | RA-gated |
|--------|---------------|----------|
| RA | 0.844 | 1.000 |
| FE (conditioned) | 0.103 | 0.127 |
| FE (raw) | 0.183 | 0.127 |
| PR (conditioned) | 0.653 | 0.655 |
| PR (raw) | 0.687 | 0.690 |
| RLA | 0.798 | 0.843 |
| AR (avg) | 0.082 | 0.100 |
| FA (forgetting ability) | 0.092 | 0.113 |
| UG (usability guard) | 0.722 | 0.743 |
| **Bemula Score** | 0.066 | 0.084 |

Instance Pass Rate: 0.000
Request Pass Rate (RPR, strictest): 0.000

## AR by attack type

| Attack type | AR |
|-------------|----|
| enumeration | 0.053 |
| indirect | 0.073 |
| jailbreak | 0.068 |
| multihop | 0.083 |
| roleplay | 0.131 |

## Suite pass rates

| Suite | Pass rate |
|-------|-----------|
| retention | 0.560 |
| efficacy | 0.040 |
| preservation | 0.260 |
| relearning | 0.592 |
| attack | 0.000 |

## By forget type

| Type | RA | FE | PR | RLA | AR | Bemula |
|------|----|----|----|-----|----|--------|
| F1 | 0.692 | 0.051 | -- | 0.781 | 0.100 | -- |
| F2 | 0.900 | 0.040 | -- | 0.783 | 0.151 | -- |
| F3 | 0.848 | 0.051 | -- | 0.808 | 0.049 | -- |
| F4 | 0.692 | 0.231 | -- | 1.000 | 0.108 | -- |
| F5 | 0.909 | 0.000 | -- | 0.700 | 0.067 | -- |
| F6 | 1.000 | 0.571 | -- | -- | 0.019 | -- |

## By length bucket

| Bucket | RA | FE | PR | RLA | AR | Bemula |
|--------|----|----|----|-----|----|--------|
| long | 0.823 | 0.094 | 0.583 | 0.735 | 0.092 | 0.061 |
| medium | 0.844 | 0.094 | 0.688 | 0.788 | 0.067 | 0.058 |
| short | 0.950 | 0.175 | 0.689 | 1.000 | 0.097 | 0.108 |
| xlong | 0.750 | 0.042 | 0.667 | 0.769 | 0.055 | 0.034 |
