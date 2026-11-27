# mlx-community/Qwen3-VL-4B-Instruct-8bit @ http://localhost:8081/v1

20261127-125409 UTC, 46 cases

```
case                       check              expect     verdict     status            ms tokens
------------------------------------------------------------------------------------------------
fall-pos-01                fall               confirmed  confirmed   ok             12067   1357  ok
fall-pos-02                fall               confirmed  confirmed   ok             11592   1356  ok
fall-pos-03                fall               confirmed  confirmed   ok             12425   1357  ok
fall-pos-04                fall               confirmed  rejected    inconsistent   12131   1295  WRONG
fall-pos-05                fall               confirmed  confirmed   ok             12133   1183  ok
fall-neg-01                fall               rejected   confirmed   ok             13832   1613  WRONG
fall-neg-02                fall               rejected   rejected    inconsistent   10020   1357  ok
fall-neg-03                fall               rejected   rejected    ok             11237   1613  ok
fall-neg-04                fall               rejected   rejected    inconsistent   11190   1613  ok
fall-neg-05                fall               rejected   rejected    inconsistent   12130   1612  ok
fight-pos-01               fight              confirmed  confirmed   ok              8384   1147  ok
fight-pos-02               fight              confirmed  confirmed   ok              8382   1064  ok
fight-pos-03               fight              confirmed  confirmed   ok              8549   1148  ok
fight-pos-04               fight              confirmed  confirmed   ok              8978   1202  ok
fight-pos-05               fight              confirmed  confirmed   ok              8132    873  ok
fight-neg-01               fight              rejected   rejected    ok              7475   1148  ok
fight-neg-02               fight              rejected   rejected    ok              7910   1149  ok
fight-neg-03               fight              rejected   rejected    ok              7016   1147  ok
fight-neg-04               fight              rejected   rejected    ok              6971   1149  ok
fight-neg-05               fight              rejected   rejected    ok              7241   1148  ok
theft-pos-01               concealment        confirmed  confirmed   ok             13445   1863  ok
theft-pos-02               concealment        confirmed  confirmed   ok             13078   1863  ok
theft-pos-03               concealment        confirmed  confirmed   ok             13433   1863  ok
theft-pos-04               concealment        confirmed  confirmed   ok             13838   1863  ok
theft-pos-05               concealment        confirmed  confirmed   ok             14030   1863  ok
theft-neg-01               concealment        rejected   rejected    ok             12016   1864  ok
theft-neg-02               concealment        rejected   rejected    ok             12196   1863  ok
theft-neg-03               concealment        rejected   confirmed   ok             13892   1863  WRONG
theft-neg-04               concealment        rejected   rejected    ok             11535   1863  ok
theft-neg-05               concealment        rejected   rejected    ok             11500   1863  ok
smoke-pos-01               smoking            confirmed  confirmed   ok              9638   1188  ok
smoke-pos-02               smoking            confirmed  confirmed   ok              9804   1188  ok
smoke-pos-03               smoking            confirmed  confirmed   ok             10048   1188  ok
smoke-pos-04               smoking            confirmed  confirmed   ok              9949   1188  ok
smoke-pos-05               smoking            confirmed  confirmed   ok             10057   1188  ok
smoke-neg-01               smoking            rejected   rejected    ok              7447   1188  ok
smoke-neg-02               smoking            rejected   rejected    ok              7654   1188  ok
smoke-neg-03               smoking            rejected   rejected    ok              8101   1188  ok
smoke-neg-04               smoking            rejected   rejected    ok              8814   1188  ok
smoke-neg-05               smoking            rejected   confirmed   ok             11195   1188  WRONG
ppe-pos-01                 helmet             rejected   confirmed   ok             10663   1132  WRONG
ppe-pos-02                 helmet             rejected   rejected    inconsistent    8980   1132  ok
ppe-pos-03                 helmet             rejected   rejected    inconsistent   10474   1132  ok
ppe-neg-01                 helmet             confirmed  confirmed   ok             10525   1132  ok
ppe-neg-02                 helmet             confirmed  confirmed   ok             11208   1132  ok
ppe-neg-03                 helmet             confirmed  confirmed   ok             10355   1132  ok
```

```json
{
  "cases": 46,
  "with_expected": 46,
  "correct": 41,
  "accuracy": 0.891,
  "alerts": {
    "real_events": 23,
    "caught": 22,
    "missed": 1,
    "false_events": 23,
    "false_alarms": 4,
    "recall": 0.957,
    "false_alarm_rate": 0.174
  },
  "statuses": {
    "ok": 40,
    "inconsistent": 6
  },
  "latency_ms_p50": 10474.3,
  "latency_ms_p95": 13837.6,
  "prompt_tokens_mean": 1366,
  "model": "mlx-community/Qwen3-VL-4B-Instruct-8bit",
  "backend": "http://localhost:8081/v1",
  "wall_s": 486.5,
  "results": "20261127-125409-mlx-community-qwen3-vl-4b-instruct-8bit.jsonl"
}
```
