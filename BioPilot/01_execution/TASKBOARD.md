# BioPilot — Task Board

> Operational task board for **BioPilot.ai**, maintained by Claude as execution assistant.
> Co-founders: **Nitsan** (software domain) · **Assaf Rotem** (scientific domain).
> Current engagement: external contractor for **Newton Tech** — training & productizing the
> foundational motion model.
>
> **Scope:** this board is **BioPilot-internal** (our context, alignment, execution between Nitsan &
> Assaf). Newton's own system of record is **ClickUp**; we may need to track Newton-facing work there
> too (see `NT-011` and `biopilot/02_reference/ISO13485_obligations.md`).
>
> Last board update: **2026-08-13**

---

## Legend & conventions

- **Columns (flow):** `Backlog` → `Next Up` → `In Progress` → `Blocked` → `Done`.
  Move a task's bullet between sections as its status changes.
- **Task line:** `- **[NT-NNN]** Title — _owner_ · <PO tag> · <priority> · updated <YYYY-MM-DD>`
  - Optional indented line below for detail, next action, or blocker.
- **IDs:** `NT-NNN` (Newton Tech), monotonically increasing, **never reused**. Next free ID: **NT-015**.
- **Owners:** `Nitsan`, `Assaf`, or both.
- **Workstream / PO tags** (per signed PO — see `biopilot/02_reference/PO_structure.md`):
  `PO 005` foundational model migration & training (parts noted `Pt1/Pt2/Pt3`) ·
  `NGP` Next-Gen Platform (proposed, no PO yet) ·
  `PO 006`/`PO 007` per CLAUDE.md but **unverified** (its PO table is stale — reconcile with Newton).
- **Priority:** `now` · `soon` · `later`.
- **Flags:** `⚠ batch-effect` (approach is batch-effect-sensitive — verify confound) ·
  `🔒 IP §5.6` (touches jointly-owned model/training pipeline — flag before external sharing).
- Completed tasks move to `Done` with a completion date.

---

## Backlog

- **[NT-006]** Nervegen pipeline productionalization & QA — _Nitsan_ · `PO 007` · later · updated 2026-08-13
  - Refactor research code into `src/` with tests; scope per PO 007 deliverable.
- **[NT-011]** Adopt ClickUp for Newton-facing engagement tracking (when required) — _Nitsan + Assaf_ · later · updated 2026-08-13
  - Newton's system of record. Confirm when/whether we operate directly in it; our internal board can feed it. See `biopilot/02_reference/ISO13485_obligations.md`.
- **[NT-012]** Migrate this repo from BioPilot's GitHub to Newton Tech's GitHub — _Nitsan_ · `PO 005` · later · 🔒 IP §5.6 · updated 2026-08-13
  - **Trigger:** training complete = **PO 005 Part 2 done** (in-house 10k pretraining). Repo currently lives under Nitsan's BioPilot GitHub account. External IP transfer of jointly-owned model/training code — flag for IP review per §5.6 before moving.
- **[NT-013]** Next-Gen Platform Part 1 — PRD & architecture definition with Yuval & Ohad — _Nitsan + Assaf_ · `NGP` · soon · updated 2026-08-13
  - Proposed (no PO yet); **no charge**. Requires BioPilot participation in the current-platform (TAU project) meeting. Deliverable: documentation + architecture design. See `biopilot/02_reference/PO_structure.md`.
- **[NT-014]** Next-Gen Platform Part 2 — implementation plan (deliverables & milestones) — _Nitsan + Assaf_ · `NGP` · later · updated 2026-08-13
  - Proposed; a future PO will be issued based on this plan. Depends on NT-013.

## Next Up

- **[NT-003]** ntds migration into in-house training env — parallel walk at scale over the single shared drive — _Nitsan_ · `PO 005` Pt2 · soon · 🔒 IP §5.6 · updated 2026-08-13
  - Per memory `ntds_migration_architecture`: one shared drive (not 21); parallel walk required at this scale.
- **[NT-004]** Batch-effect mitigation strategy — decide architecture-level vs ML-head vs preprocessing — _Assaf + Nitsan_ · soon · ⚠ batch-effect · updated 2026-08-13
  - Dominant technical problem; Romberg / sit-to-stand most affected, treadmill leaks too.
- **[NT-007]** Own the training procedure — end-to-end training pipeline for the foundational model — _Assaf_ · `PO 005` Pt2 · now · 🔒 IP §5.6 · updated 2026-08-13
  - Assaf owns training methodology; depends on AWS infra (NT-008) being ready.
- **[NT-009]** Read & understand essential ISO 13485 presentation "ISO general presentation_Aug_2026" (Drive → QMS → Training presentations) — _Assaf + Nitsan_ · now · **due 2026-08-19** · updated 2026-08-13
  - ISO 13485 compliance flows down from Newton (their QMS). See `biopilot/02_reference/ISO13485_obligations.md`. Each of us emails VP R&D individually (no reply-all) when done; they then send a form to sign (NT-010).
- **[NT-010]** Sign ISO 13485 training acknowledgment form — _Assaf + Nitsan_ · now · updated 2026-08-13
  - Depends on NT-009 (VP R&D sends the form after our confirmation emails). Each founder signs personally.

## In Progress

- **[NT-001]** Local validation of foundational-model forward pass — _Assaf_ · `PO 005` Pt1 · now · 🔒 IP §5.6 · updated 2026-08-13
  - Assaf validating forward pass locally (incl. age/BMI/gender ML head vs prior model). From commit `cdfd76f`.
- **[NT-002]** In-house ntds embedding + metadata-probe pipeline (Fig-2 reproduction) — _Nitsan_ · `PO 005` Pt1 · now · 🔒 IP §5.6 · updated 2026-08-13
  - From commit `6f021e7`.
- **[NT-008]** AWS infrastructure setup to facilitate training — _Nitsan_ · `PO 005` Pt2 · now · 🔒 IP §5.6 · updated 2026-08-13
  - Standing up compute/storage so Assaf's training procedure (NT-007) can run at scale.

## Blocked

- **[NT-005]** ~300 videos from third company — intake & processing — _Nitsan_ · later · updated 2026-08-13
  - Blocked: pending funding (`on-funding`).

## Done

- **[NT-000]** ntds demographics metadata gap resolved — 56% → 98.9% coverage via `rescrape_10k_metadata.py` — _Nitsan_ · `PO 005` Pt2 · completed 2026-08-07
  - Per memory `ntds_migration_metadata_gap`.
