# Stationary laboratory protocol for supervisor review

**Primary question.** At matched event recall, does personalized calibration with quality gating reduce false alert onsets per observed hour? Pre-register minimum acceptable coverage, operating points and latency before opening the final test partition.

**Collection.** Obtain institutional review before recruitment. Use a parked laboratory rig or simulator. Assign anonymous participant, session and observer codes. Record lighting, eyewear, pose, camera and software revision. Begin facing forward with eyes comfortably open for five seconds. Counterbalance natural blinking, talking, simulated prolonged closure, yawning, head rotation and partial obstruction. Test cable failure separately. Stop on discomfort.

**Reference labels.** An independent observer marks onset and offset on the session clock with predictions hidden. Two observers annotate a subset. Adjudicate disagreements before unlocking predictions. Store corrections as new rows referencing the earlier annotation. Keep unsuccessful calibrations and unavailable intervals in the operational denominator. Simulated eye closure and self-reported vigilance are separate outcomes.

**Consent text for adaptation.** I understand the purpose, tasks, expected duration, possible discomfort, voluntary nature and withdrawal procedure explained by the investigator. I understand which measurements will be retained, who may access them and for how long. I may skip any task or stop without penalty. I give separate permission for recording or publication of identifiable images if applicable. Investigator supplies the institutional contact, retention period, withdrawal deadline and approval identifier before consent. Signed forms remain private.

**Analysis.** Freeze participant splits and reference labels first. Review recuts and near duplicates manually. SHA-256 detects exact duplicates only. Report counts by subject and condition, calibration failures, valid time, full-session coverage and eligible-only metrics. Resample whole participants for paired confidence intervals. Match at most one onset within each annotated event. Extra onsets count as false alerts. Report hardware alarms for tracking loss separately.

**Hardware.** Isolate actuators on a stationary rig. Confirm port identity, then exercise ASCII 0, 1, 2, exit 3, loss/reconnect, and USB unplug. Independently timestamp receipt and the physical indication. The host serial write time measures a different interval.

## Data sources and adapters

| Source | Acquisition | Annotation unit |
|---|---|---|
| [UTA-RLDD](https://sites.google.com/view/utarldd/home) | Use the authors' dataset link. Keep five official participant folds and per-person publication permissions. | Predominant self-reported video class 0, 5 or 10. Aggregate features once per video. |
| [YawDD](https://doi.org/10.21227/e1qm-hb90) | Follow the IEEE DataPort access link and retain accompanying terms and image permissions. | Independently mark yawn intervals. Include talking as a control. |
| [NTHU-DDD](https://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/) | Request the released data and licence from NTHU CV Lab. Their page timed out during verification. Confirm the release-specific labels and official split before conversion. | Convert independently supplied frame annotations to half-open intervals using video timestamps. |

CSV manifest columns are dataset,video,participant,session,split,label,annotation. Paths are relative to the manifest. Dataset names are uta-rldd, yawdd, nthu or lab. Splits are development/test. For interval collections supply annotation CSVs with start,end,label in seconds. For UTA use the video label and leave annotation empty. Keep each dataset in a separate extraction. The interval trainer rejects video-level labels. No public dataset was downloaded in this run.
