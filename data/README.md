# Fixed test files

Each task folder contains only `test.jsonl`: 250 fixed questions from the supplied experiment packages. No new split is made and no dataset download occurs at evaluation time. The evaluator reads no training or validation files.

`manifest.json` records source archive/member, original count, and SHA-256. Competition Math's first 250 original test rows exactly match the later 250-question run's recorded hash `b0c48da13bc060088ffec6cc07eff80f69113e6d486130f0d027b28e87e71549`.

These files establish which questions this package evaluates. They do not establish whether a teammate's checkpoint previously saw any of those questions. Confirm held-out test identity with each model owner before making an SFT/GRPO comparison.

| Task | Model input | Grading fields |
| --- | --- | --- |
| KodCode | `question`, optionally published function declarations from `test_info` | `test`; `solution` for optional reference checks |
| OpenCode | `input` | `unit_tests`; `output` for optional reference checks |
| Competition Math | `problem` with the fixed task instruction | Boxed answer in `solution` |
| Shortest Path | `system` and `user` | `matrix`; `answer` checked independently using shortest-path search |

References, test bodies, and gold answers never form part of the built-in model prompts. Original metadata remains in the files so provenance and grading can be audited. Shortest Path IDs are derived from the matrix content; Competition Math IDs are derived from the problem content. Original KodCode/OpenCode IDs are retained.
