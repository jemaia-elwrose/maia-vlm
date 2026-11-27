# Qwen/Qwen3-VL-4B-Instruct @ http://localhost:8001/v1

20261127-112011 UTC, 46 cases

```
case                       check              expect     verdict     status            ms tokens
------------------------------------------------------------------------------------------------
fall-pos-01                fall               confirmed  confirmed   ok              7637   1357  ok
fall-pos-02                fall               confirmed  confirmed   ok              7327   1356  ok
fall-pos-03                fall               confirmed  confirmed   ok              7600   1357  ok
fall-pos-04                fall               confirmed  rejected    inconsistent    7598   1295  WRONG
fall-pos-05                fall               confirmed  confirmed   ok              7956   1183  ok
fall-neg-01                fall               rejected   confirmed   ok              8646   1613  WRONG
fall-neg-02                fall               rejected   rejected    inconsistent    5959   1357  ok
fall-neg-03                fall               rejected   rejected    inconsistent    8781   1613  ok
fall-neg-04                fall               rejected   rejected    inconsistent    6357   1613  ok
fall-neg-05                fall               rejected   rejected    inconsistent    7014   1612  ok
fight-pos-01               fight              confirmed  confirmed   ok              4680   1147  ok
fight-pos-02               fight              confirmed  confirmed   ok              4917   1064  ok
fight-pos-03               fight              confirmed  confirmed   ok              4980   1148  ok
fight-pos-04               fight              confirmed  confirmed   ok              5027   1202  ok
fight-pos-05               fight              confirmed  confirmed   ok              4559    873  ok
fight-neg-01               fight              rejected   rejected    ok              3006   1148  ok
fight-neg-02               fight              rejected   rejected    ok              3104   1149  ok
fight-neg-03               fight              rejected   rejected    ok              2860   1147  ok
fight-neg-04               fight              rejected   rejected    ok              2944   1149  ok
fight-neg-05               fight              rejected   rejected    ok              3110   1148  ok
theft-pos-01               concealment        confirmed  confirmed   ok              6500   1863  ok
theft-pos-02               concealment        confirmed  confirmed   ok              6068   1863  ok
theft-pos-03               concealment        confirmed  confirmed   ok              6170   1863  ok
theft-pos-04               concealment        confirmed  confirmed   ok              6533   1863  ok
theft-pos-05               concealment        confirmed  confirmed   ok              6399   1863  ok
theft-neg-01               concealment        rejected   rejected    ok              4835   1864  ok
theft-neg-02               concealment        rejected   rejected    ok              5061   1863  ok
theft-neg-03               concealment        rejected   rejected    ok              5586   1863  ok
theft-neg-04               concealment        rejected   rejected    ok              4495   1863  ok
theft-neg-05               concealment        rejected   rejected    ok              4514   1863  ok
smoke-pos-01               smoking            confirmed  confirmed   ok              4751   1188  ok
smoke-pos-02               smoking            confirmed  confirmed   ok              4752   1188  ok
smoke-pos-03               smoking            confirmed  confirmed   ok              5089   1188  ok
smoke-pos-04               smoking            confirmed  confirmed   ok              4877   1188  ok
smoke-pos-05               smoking            confirmed  confirmed   ok              4757   1188  ok
smoke-neg-01               smoking            rejected   rejected    ok              2563   1188  ok
smoke-neg-02               smoking            rejected   rejected    ok              2768   1188  ok
smoke-neg-03               smoking            rejected   rejected    ok              3388   1188  ok
smoke-neg-04               smoking            rejected   rejected    ok              3015   1188  ok
smoke-neg-05               smoking            rejected   rejected    ok              3461   1188  ok
ppe-pos-01                 helmet             rejected   confirmed   ok              5713   1132  WRONG
ppe-pos-02                 helmet             rejected   rejected    ok              5925   1132  ok
ppe-pos-03                 helmet             rejected   rejected    inconsistent    5410   1132  ok
ppe-neg-01                 helmet             confirmed  confirmed   ok              5482   1132  ok
ppe-neg-02                 helmet             confirmed  confirmed   ok              6056   1132  ok
ppe-neg-03                 helmet             confirmed  confirmed   ok              5318   1132  ok
```

```json
{
  "cases": 46,
  "with_expected": 46,
  "correct": 43,
  "accuracy": 0.935,
  "alerts": {
    "real_events": 23,
    "caught": 22,
    "missed": 1,
    "false_events": 23,
    "false_alarms": 2,
    "recall": 0.957,
    "false_alarm_rate": 0.087
  },
  "statuses": {
    "ok": 40,
    "inconsistent": 6
  },
  "latency_ms_p50": 5061.0,
  "latency_ms_p95": 7956.2,
  "prompt_tokens_mean": 1366,
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "backend": "http://localhost:8001/v1",
  "wall_s": 266.5,
  "results": "20261127-112011-qwen-qwen3-vl-4b-instruct.jsonl"
}
```
