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
- **IDs:** `NT-NNN` (Newton Tech), monotonically increasing, **never reused**. Next free ID: **NT-012**.
- **Owners:** `Nitsan`, `Assaf`, or both.
- **Workstream / PO tags** (per CLAUDE.md): `PO 005` recording · `PO 006` Solid ·
  `PO 007` Nervegen · `PO 008` model migration.
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

## Next Up

- **[NT-003]** ntds migration into in-house training env — parallel walk at scale over the single shared drive — _Nitsan_ · `PO 008` · soon · 🔒 IP §5.6 · updated 2026-08-13
  - Per memory `ntds_migration_architecture`: one shared drive (not 21); parallel walk required at this scale.
- **[NT-004]** Batch-effect mitigation strategy — decide architecture-level vs ML-head vs preprocessing — _Assaf + Nitsan_ · soon · ⚠ batch-effect · updated 2026-08-13
  - Dominant technical problem; Romberg / sit-to-stand most affected, treadmill leaks too.
- **[NT-007]** Own the training procedure — end-to-end training pipeline for the foundational model — _Assaf_ · `PO 008` · now · 🔒 IP §5.6 · updated 2026-08-13
  - Assaf owns training methodology; depends on AWS infra (NT-008) being ready.
- **[NT-009]** Read & understand essential ISO 13485 presentation "ISO general presentation_Aug_2026" (Drive → QMS → Training presentations) — _Assaf + Nitsan_ · now · **due 2026-08-19** · updated 2026-08-13
  - ISO 13485 compliance flows down from Newton (their QMS). See `biopilot/02_reference/ISO13485_obligations.md`. Each of us emails VP R&D individually (no reply-all) when done; they then send a form to sign (NT-010).
- **[NT-010]** Sign ISO 13485 training acknowledgment form — _Assaf + Nitsan_ · now · updated 2026-08-13
  - Depends on NT-009 (VP R&D sends the form after our confirmation emails). Each founder signs personally.

## In Progress

- **[NT-001]** Local validation of foundational-model forward pass — _Assaf_ · `PO 008` · now · 🔒 IP §5.6 · updated 2026-08-13
  - Assaf validating forward pass locally (incl. age/BMI/gender ML head vs prior model). From commit `cdfd76f`.
- **[NT-002]** In-house ntds embedding + metadata-probe pipeline (Fig-2 reproduction) — _Nitsan_ · `PO 008` · now · 🔒 IP §5.6 · updated 2026-08-13
  - From commit `6f021e7`.
- **[NT-008]** AWS infrastructure setup to facilitate training — _Nitsan_ · `PO 008` · now · 🔒 IP §5.6 · updated 2026-08-13
  - Standing up compute/storage so Assaf's training procedure (NT-007) can run at scale.

## Blocked

- **[NT-005]** ~300 videos from third company — intake & processing — _Nitsan_ · later · updated 2026-08-13
  - Blocked: pending funding (`on-funding`).

## Done

- **[NT-000]** ntds demographics metadata gap resolved — 56% → 98.9% coverage via `rescrape_10k_metadata.py` — _Nitsan_ · `PO 008` · completed 2026-08-07
  - Per memory `ntds_migration_metadata_gap`.
