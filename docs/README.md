# docs/

Reviewers clone this repository and read this folder first, so the required PDFs
must live here under exactly these names.

| File | Due | Content |
| --- | --- | --- |
| `proposal.pdf` | Thu 01.10.2026, 23:59 | Max. 2 pages. Problem statement · originality & motivation · data source & features · system design (with the FTI diagram embedded). Also submitted on ILIAS. |
| `ms2_summary.pdf` | Thu 05.11.2026, 23:59 | Max. 2 pages. What MS2 achieved + a numbered response to every MS1 review point. |
| `ms3_summary.pdf` | Thu 03.12.2026, 23:59 | Max. 2 pages. What MS3 achieved + a numbered response to every MS2 review point. |
| `ms4_summary.pdf` | Sat 10.01.2027, 23:59 | Max. 2 pages. What changed in MS4 + a numbered response to every MS3 review point. |

The repository is graded at the **last commit before each deadline**, so commit
the PDF before 23:59 — not the morning after.

## Working files kept here

| File | Purpose |
| --- | --- |
| `architecture.svg` | The FTI diagram. Embedded in the root `README.md`; embed it in `proposal.pdf` too (the proposal must not use an ASCII diagram). |
| `proposal.md` | Working draft of the proposal. Edit it, then export to `proposal.pdf`. |
| `milestone_summary_template.md` | Structure for each `ms<n>_summary.pdf`, including the response-to-reviewers format. |

## Exporting a draft to PDF

Any Markdown-to-PDF route is fine. With Pandoc:

```bash
pandoc docs/proposal.md -o docs/proposal.pdf \
  --pdf-engine=xelatex -V geometry:margin=2.5cm
```

Check the result is **at most 2 pages** and that the diagram is legible.

## Peer reviews

Reviews are uploaded to ILIAS, not committed here. They are anonymous: the file
must carry no name anywhere, **including the PDF metadata** (check File →
Properties before uploading). Naming: `MS<n>_P<projectID>_Reviewer<A|B>.pdf`.
