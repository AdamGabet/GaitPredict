# Newton Tech ISO 13485 — Obligations Flowing Down to BioPilot

> **Source:** Newton Tech VP R&D (owns the QMS process), email following a Google Meet on **2026-08-12**.
> **Why this applies to us:** Newton Tech is ISO 13485-certified. ISO 13485 covers *all* of Newton's
> activities — explicitly including **research & development**, which is our contracted scope. As
> Newton's external R&D contractor, our deliverables and documented activities fall under their QMS,
> so these obligations flow down to BioPilot (Nitsan & Assaf).
>
> This doc restates the transferred items as **standing obligations for how we work**. Time-bound
> actions are tracked on the board (`biopilot/01_execution/TASKBOARD.md`) as `NT-009` / `NT-010`.

---

## Core values (as stated by Newton)

1. **Safety and quality above all** — of products, processes, and our execution.
2. Implement the **agreed procedures**.
3. All documents follow our processes: **available to relevant leaders, maintained, version-controlled, signed**.
4. Maintain **traceability between documents**.
5. **Measure as much as we can** — OKRs, V&V, bugs, statistics.
6. **Constant improvement** — via measurement, lessons-learned, customer feedback, training, knowledge transfer, design reviews, transparency.
7. **Reduce risk**, especially safety-related risk.
8. **All testing is always documented.**
9. All activities done with **customer satisfaction** in mind.
10. File data/knowledge in the **electronic tools**: Drive / GitHub / ClickUp / Chat.
11. **Cybersecurity and customer privacy** are critical.

> **ClickUp** is Newton's tool for internal task management (confirmed). It is Newton's **system of
> record**; BioPilot will likely need to work in it at some point (tracked as `NT-011`). This is
> distinct from our internal board — see below.

## Our board vs. Newton's ClickUp

- `biopilot/01_execution/TASKBOARD.md` is **BioPilot-internal**: our own context, alignment, and
  execution tracking between Nitsan & Assaf.
- **ClickUp** is **Newton's** system of record for the engagement. When required, we track/report
  Newton-facing work there per their QMS; our internal board can mirror or feed it but is not a
  substitute for it.

## Documentation model

- Levels: general presentations, procedures, checklists, templates, documents across disciplines, reports, statistics.
- All documents kept and maintained in **electronic frameworks**, with **version control, signatures, and controlled availability**.
- **Traceability** maintained between related documents.
- Templates and example documents live in **Drive** — use them as the basis for our future documented activities.

## What this means for BioPilot's day-to-day (standing practices)

These reinforce and extend the existing standards in `CLAUDE.md`:

- **Version control & traceability:** work in GitHub with meaningful history; link artifacts (code → results → reports) so provenance is traceable. Flag `🔒 IP §5.6` items before external sharing.
- **Documented testing:** every V&V / test run is documented — what was tested, inputs, outputs, pass/fail. No undocumented validation.
- **Measurement:** capture metrics wherever feasible (model V&V results, bug counts, KPIs) with CIs on reported metrics, per our model-dev standards.
- **Risk reduction:** name safety- and reproducibility-relevant risks proactively; flag batch-effect exposure (`⚠ batch-effect`) on any cross-site/cross-session analysis.
- **Cybersecurity & privacy:** never commit patient data/video/PII; anonymized IDs only; secrets in env/secret manager. (Already in `CLAUDE.md` — now also an ISO obligation.)
- **Use Newton's electronic tools** (Drive / GitHub / ClickUp / Chat) as the systems of record; use Drive templates for documented deliverables.
- **Improvement loop:** lessons-learned, design reviews, and knowledge transfer between Nitsan & Assaf are expected, documented practices — not optional.

## Immediate action items (tracked on the board)

- **Read & understand** the essential presentation **"ISO general presentation_Aug_2026"**
  (Drive → QMS folder → Training presentations). Deadline: **within ~1 week of the 2026-08-12 meeting → by 2026-08-19**. → `NT-009`
- **Each** of us (Nitsan & Assaf) emails the VP R&D confirming completion — **individually, no reply-all**.
- VP R&D then sends a **short acknowledgment form to sign**. → `NT-010`
  - Note: signing is a personal action each founder performs themselves.

## Open items to confirm with VP R&D

- Location/name of the templates we're expected to adopt for our deliverables.
- When/whether we're expected to operate directly in Newton's ClickUp for engagement tracking.
