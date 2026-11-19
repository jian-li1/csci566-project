# Verification

- 18 CPU checks passed on the authoring machine (Python 3.14).
- The four supplied test files each contain 250 questions and match the original experiment SHA-256 hashes.
- All 1,000 task prompts rendered with a tokenizer stub; the tests check that reference answers and unit-test bodies are excluded. Native tokenizers still need the teammate's smoke run.
- All 250 Shortest Path references passed the actual reference-check entry point using independent graph search, without loading a model.
- Both code-worker interfaces passed fixed, authored addition-function probes. Model-generated code was not executed on the host.
- An interrupted evaluation resumed without repeating completed predictions; changed generation settings were rejected.

GPU inference, real adapter attachment, symbolic Math comparison dependencies, and Docker/Apptainer image execution were not available for verification here. Run the documented five-question smoke evaluation with the actual checkpoint and agreed grader before the full test run.

No training or validation files or model weights are included in this package.
