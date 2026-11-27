# Verification and research readiness

## Executed on Windows, Python 3.10.9

| Command or workflow | Observed result |
|---|---|
| python -m unittest discover -s tests -v | Ran 18 tests in 7.169 s. OK. |
| python -m compileall -q monitor tests | Exit 0. |
| node --check on extracted dashboard JavaScript | Exit 0. |
| python -m monitor.run --headless --video runs/blank20.avi --db runs/verified.sqlite3 | 20 decoded frames, 20 stored observations, 20 unknown fatigue states, zero dropped records. |
| Real HTTP requests to a loopback Uvicorn server | Dashboard and health served. Stale session cleared its state. Annotation creation returned 201, invalid interval returned 422, history and CSV returned the stored data. |
| Legacy partial flush | All seven records persisted to SQLite and JSONL on shutdown in an isolated fixture. |
| Four supervised model families and four feature subsets | Grouped training, held-out scoring, model serialization, reload and inference exercised using a unit-test fixture. Its scores carry no scientific interpretation. |
| Serial worker with mock transport | A sustained critical state emitted a single 2 and shutdown emitted 3. |
| Historical JSONL inventory | All eight files parsed. 11,983 telemetry rows in seven nonempty files. |

The short video fixture comprises 128 x 128 uniform frames at 10 encoded FPS. It was decoded as fast as possible. Median post-read processing time was 2.375 ms and final processing throughput was 366.91 frames/s in this particular run. This measures the no-face replay path. It provides no evidence about webcam FPS, face-present landmark throughput or physical alert latency.

## Dependency checks

The host had MediaPipe 0.10.9 with an incompatible protobuf 7.36.1. The original pip check reported the conflict. A normal pip install could not resolve the package host from this workspace. For the tests, project-independent temporary source copies supplied FastAPI 0.115.12, Pydantic 1.10.24, Starlette 0.46.2 and the protobuf 3.20.3 pure-Python runtime with generated descriptor modules. OpenCV 4.13.0, NumPy 2.2.6 and scikit-learn 1.7.2 were installed locally. No third-party source workaround is included in the application. A clean wheel installation of the exact requirements on Windows and Linux remains to be verified. The suite used unittest, which is also discoverable by pytest.

## Evidence still required

- No independent participant labels were present in the historical logs. Accuracy, Macro F1, class recall, event latency, ablation gain and false alarms per hour on real subjects are unavailable. All improvement hypotheses remain open.
- External dataset acquisition and release-specific annotation conversion require a licensed copy. The manifest adapters implement extraction and granularity checks. They have not been validated against the full official distributions. The interval trainer does not yet evaluate UTA video-level summaries.
- Real camera tracking, calibrated pose error, occlusion recall, calibration reliability across faces and reconnect behaviour require laboratory measurements. Camera-driver frame losses cannot be inferred from the capture requests.
- The serial transport has no documented ACK protocol or firmware pin map. Physical indication, latency and long-duration stability require the stationary hardware procedure.
- Complete the planned fixed-EAR and original-adaptive benchmark comparisons, quality-gate and personalization ablations, near-duplicate review, condition-wise uncertainty estimates, and matched-coverage analysis before interpreting efficacy.
- Current model selection uses development macro F1. Operational constraints, hyperparameter selection and an independently reviewed final operating point remain open. The runtime reports the optional classifier separately from the hardware closure rule.
- The dashboard currently supports monitoring, annotations, historical sessions, exports and held-out report summaries. Full browser interaction, multi-operator correction conflicts, persistent profile management and the full experiment-management workflow remain to be verified or extended. Raw video is not recorded by default.
- The IEEE-style source and formatted reading copy are methods drafts. Confirm authors, approvals, data availability terms and research conclusions after experiments. LaTeX compilation against IEEEtran was unavailable here.
