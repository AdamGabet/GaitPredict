# Newton-Tech × BioPilot — PO & Workstream Structure

> **Source:** signed proposal "Newton-Tech × BioPilot — Proposed Next Work (PO 005)".
> **Note:** this supersedes the PO table in `CLAUDE.md`, which is **stale** — it lists PO 005 as
> "recording". The actual signed **PO 005 is foundational model migration & training** (below).
> The `CLAUDE.md` PO 006/007/008 rows should be reconciled with Newton too — treat them as unverified.

---

## PO 005 — Movement Intelligence: Foundational model migration & training  *(signed)*

**Scope.** Migrate our soon-to-be-published, co-authored motion model into Newton's in-house
environment and re-train it on the 10k dataset. Prove Newton-Tech can train its own model so it can
build proprietary models on its proprietary data in future. 🔒 IP §5.6 (jointly-owned model/training).

| Part | $ / duration | Work | Deliverable |
|------|-------------|------|-------------|
| **Part 1** | $5k / 2 wk | Model migration & in-house deployment: migrate the co-authored model into Newton's env, verify it runs end-to-end | Working in-house model, integration notes + source code |
| **Part 2** | $5k / 2 wk | **Pretraining with 10k data** using the original training setup — establishes Newton can train in-house on its own infra | In-house-trained model + training pipeline |
| **Part 3** | $5k / 2 wk | Pretraining optimization: tune masking, augmentation, loss weighting, optimizer schedule, embedding pooling so in-house model matches the published model | Validated training recipe Newton reuses on its own data |

**"Training complete" = PO 005 Part 2 done** (in-house pretraining on 10k). This is the trigger for the
repo migration to Newton's GitHub (`NT-012`).

## Next-Gen Newton Tech Platform — definition & scoping  *(proposed, no PO yet)*

Current platform is customer-specific and lacks out-of-the-box scalability. Proposal: a next-gen
**distributed** platform, defined and built in close collaboration with **Yuval and Ohad**.

- **Part 1 — Product Requirements & Architecture Definition.** Define specs with Yuval & Ohad;
  deliver documentation + architecture design. **No charge.** *Requires BioPilot team participation in
  the meeting discussing the current platform (i.e. the TAU project).*
- **Part 2 — Platform Implementation Plan.** Execution plan with deliverables & milestones for the
  defined scope. **A future PO will be issued based on this plan.**

---

## Board tag mapping

- `PO 005` on the board → this workstream; part noted as `(Pt1/Pt2/Pt3)` where determinable.
  Part assignments of existing tasks are BioPilot's best mapping — confirm against Newton's framing.
- `NGP` → Next-Gen Platform (proposed).
