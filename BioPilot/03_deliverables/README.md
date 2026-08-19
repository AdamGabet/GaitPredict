# Deliverables — Document Register

Controlled deliverables for the Newton Tech engagement. Each entry is filed here as the
GitHub system-of-record copy (ISO 13485 document control); signatures are applied on the
Newton-side copies manually.

| Doc ID | Deliverable | PO / Part | Version | Date issued | Status | Files |
|--------|-------------|-----------|---------|-------------|--------|-------|
| BP-PO008-001 | Part 1 — Model Migration & In-House Deployment | PO 008 · Part 1 | 1.0 | 2026-08-13 | Draft, for Newton review | `Part1_Gait_Model_Migration.pptx`, `Part1_Gait_Model_Migration.docx` |

## BP-PO008-001 — notes

- **What it covers:** migration of the foundational gait model into BioPilot's in-house
  environment and an end-to-end verification run reproducing the published age / BMI / sex
  result (3,643 participants) at parity with the paper.
- **Traceability:** the reproduced pipeline is committed at `6f021e7` (in-house NTDS
  embedding + metadata-probe) and the guide at `5e550e4`.
- **IP:** the model and training pipeline are jointly-owned under master agreement §5.6 —
  flag for IP review before any external sharing.
- **Data handling:** reports contain cohort-level metrics only; no patient IDs, no raw
  recordings, no PII.
- **Approval:** to be signed by Newton (VP R&D) on their copy; the signature block is in
  both documents.
