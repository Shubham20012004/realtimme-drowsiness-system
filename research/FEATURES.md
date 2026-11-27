# Feature contract, geometry-time-v1

Image coordinates are floating-point pixels. EAR and MAR are dimensionless. Invalid geometry abstains. Camera intrinsics use the image centre and focal length equal to image width. These are approximate pose estimates.

| Field | Definition | Unit or range |
|---|---|---|
| left, right | Sum of two upper-to-lower lid distances divided by twice the corner distance | nonnegative ratio |
| normalized | Mean of left and right EAR divided separately by their calibration medians | nonnegative ratio |
| mar | Distance between lip points 13 and 14 divided by distance between corners 61 and 291 | nonnegative ratio |
| closure_s, max_closure_s | Duration of current uninterrupted closure, maximum of current and recently ended closures | seconds |
| perclos | Duration of intervals with both endpoints below 0.72 normalized EAR divided by valid duration, trailing 30 seconds | 0–1, unavailable below 70% coverage |
| coverage | Eligible adjacent-observation seconds divided by 30. Maximum adjacent gap 0.3 seconds | 0–1 |
| blink_per_min, mean_blink_s | Ended closures lasting 0.06 to under 0.8 s per observed minute, and mean duration | events/min, seconds |
| prolonged_closures | Ended closures lasting at least 0.8 s within the window | count |
| yawn_s, yawns_per_min | Duration of mouth opening above max(0.35,baseline+0.20); count after 1 s persistence | seconds, candidate events/min |
| pitch, yaw, roll | Approximate Euler angles from six landmarks and generic face geometry | degrees |
| nods_per_min | Candidate pitch excursions above baseline +12 degrees for 0.25 s | events/min |
| pitch_velocity_deg_s | Backward difference across contiguous valid pose observations | degrees/s |
| brightness, blur | Mean gray intensity and variance of Laplacian inside face bounding box | 0–255, squared intensity |
| reprojection_error | Mean six-point reprojection residual divided by mouth width | nonnegative ratio |
| actual_fps, processing_ms | Completed valid input reads divided by elapsed wall time, and per-frame post-read processing duration | frames/s, milliseconds |

Observation loss resets active events. Long gaps are excluded from the denominator. Eye-closure estimates are behavioural proxies. All thresholds are provisional development settings. Capture-driver drops and true exposure-to-actuator latency require external timing.
