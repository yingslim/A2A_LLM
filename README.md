Client → Orchestrator (LangGraph)
   ingress ─→ defense_gate ─→ router ─→ executor ─→ aggregate ─→ END
                  │              │            │
                  │              │            └─ calls victim model service(s)
                  │              └─ selects targets (single/ensemble)
                  └─ returns Decision: {allow | rewrite | block}
