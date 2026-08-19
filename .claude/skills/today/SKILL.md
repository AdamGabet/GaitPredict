---
name: today
description: Personalized daily task digest for a BioPilot co-founder (Nitsan or Assaf), built from the execution task board. Use when someone runs /today or asks for their daily digest, "what's on my plate", or a focus list for the day.
---

# /today — Personalized task digest

Produce a focused daily digest for one BioPilot co-founder from the execution task board.

## Who is this for

- Argument selects the person: `/today nitsan` or `/today assaf` (case-insensitive; also accepts
  "Assaf Rotem").
- **No argument → default to Nitsan** (this is Nitsan's environment; user email `nitsanc@gmail.com`).
- If the argument doesn't match a known co-founder, ask which one rather than guessing.

## Source of truth

Read the board: `biopilot/01_execution/TASKBOARD.md`. Do not invent tasks — everything in the digest
must trace to a task there (by `NT-NNN` ID). If the board is missing or empty, say so.

## What to include (in this order)

1. **🔴 In Progress — yours.** Tasks in the `In Progress` section owned by this person (owner is
   them or "X + Y" including them). Lead with these; this is today's real work.
2. **🚧 Blocked — yours.** Any `Blocked` task they own, with the blocker reason. Flag if a blocker is
   something they can now unblock.
3. **⏭ Next Up — yours.** `Next Up` tasks they own, ordered by priority (now → soon → later). Call
   out dependencies (e.g. "NT-007 waits on NT-008").
4. **🤖 Claude can take these.** Scan this person's active tasks (In Progress + Next Up) and surface
   the ones **Claude can meaningfully move without the owner doing the work** — drafting code, docs,
   or analysis; research; scaffolding; writing tests; preparing a plan or a draft email. For each,
   say whether it's **fully doable by Claude** or **needs input**, and if input is needed name exactly
   what (a decision, a credential the owner must supply, a file/path, an approval). Exclude what is
   inherently the owner's to do — signing forms, provisioning or paying for infra, sending messages on
   their behalf, physical actions, or a judgment call. End each with a concrete offer
   ("say the word and I'll draft X"). Omit the section if nothing qualifies.
5. **🤝 Shared / handoffs.** Tasks co-owned with the other founder, or where one person's task blocks
   the other's — surface the coordination point explicitly.
6. **📋 On deck (brief).** One-line mention of their `Backlog` items, if any. Keep terse.

Omit any section that has no matching tasks — don't print empty headers.

## Format

- Start with a one-line header: `**Today — <Name>** · <today's date> · <N active tasks>`.
- Under each section, one bullet per task: `**[NT-NNN]** Title — <priority> <flags>` then a short
  "next action" clause if one is obvious from the task note.
- Preserve flags (`⚠ batch-effect`, `🔒 IP §5.6`) — they matter for how the work is handled.
- End with a single **Focus** line: the one task you'd tell them to move first today, and why.
- Keep it scannable — this is a morning glance, not a status report. No preamble, no filler.

## Notes

- Read-only: `/today` never edits the board. If the person mentions a status change while reading
  their digest, offer to update the board as a follow-up.
- Dates on the board are `YYYY-MM-DD`; use the current date for the header.
