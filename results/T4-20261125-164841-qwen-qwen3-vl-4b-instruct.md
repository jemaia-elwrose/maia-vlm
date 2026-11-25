# Qwen/Qwen3-VL-4B-Instruct @ http://localhost:8001/v1

20261125-164841 UTC, 23 cases

```
case                       check              expect     verdict     status            ms tokens
------------------------------------------------------------------------------------------------
queue-describe             describe           -          confirmed   ok              2618    737  
queue-person               person-present     confirmed  confirmed   ok              4081    945  ok
queue-fall-x               fall               rejected   unclear     ok              3037   1155  no alert
queue-fight-x              fight              rejected   rejected    ok              2653   1148  ok
gate-describe              describe           -          confirmed   ok              2735    737  
gate-helmet                helmet             confirmed  confirmed   ok              4376    721  ok
shop-describe              describe           -          confirmed   ok              2515    737  
shop-phone                 phone-use          confirmed  confirmed   ok              3108   1013  ok
shop-smoking-x             smoking            rejected   rejected    ok              2965   1188  ok
shop-concealment-x         concealment        rejected   rejected    ok              4250   1835  ok
smoke-describe             describe           -          confirmed   ok              1809    609  
smoke-smoking              smoking            confirmed  confirmed   ok              4695   1188  ok
smoke-phone-x              phone-use          rejected   rejected    ok              3328   1013  ok
fall-describe              describe           -          confirmed   ok              2501    609  
fall-fall                  fall               confirmed  confirmed   ok              4795    964  ok
fight-describe             describe           -          confirmed   ok              2436    737  
fight-fight                fight              confirmed  confirmed   ok              4579   1149  ok
theft-describe             describe           -          confirmed   ok              2256    849  
theft-concealment          concealment        confirmed  confirmed   ok              5977   1863  ok
forklift-describe          describe           -          confirmed   ok              2687    737  
forklift-proximity         forklift-proximity confirmed  confirmed   ok              5057   1128  ok
ppe-describe               describe           -          confirmed   ok              2574    737  
ppe-helmet                 helmet             rejected   rejected    ok              4860   1066  ok
```

```json
{
  "cases": 23,
  "with_expected": 14,
  "correct": 13,
  "accuracy": 0.929,
  "alerts": {
    "real_events": 8,
    "caught": 8,
    "missed": 0,
    "false_events": 6,
    "false_alarms": 0,
    "recall": 1.0,
    "false_alarm_rate": 0.0
  },
  "statuses": {
    "ok": 23
  },
  "latency_ms_p50": 3036.6,
  "latency_ms_p95": 5056.6,
  "prompt_tokens_mean": 994,
  "model": "Qwen/Qwen3-VL-4B-Instruct",
  "backend": "http://localhost:8001/v1",
  "wall_s": 87.2,
  "results": "20261125-164841-qwen-qwen3-vl-4b-instruct.jsonl"
}
```
