# ADR 0001: Low-observer-effect benchmark contract

- Status: Proposed
- Specification revision: 4479bcc19db72f6ad243a87e4b7271496d60d0b7
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/baselines/observer-effect-v1.yaml`

## Context

SIA must not claim a low observer effect from implementation inspection, an unexecuted benchmark schema, failure to find a statistically significant difference, or measurements made on hardware unlike the claimed target. This ADR preregisters the experiment and analysis required before such a claim is permitted.

This document is proposed. Approval fixes the workload, endpoint, margin, sampling, exclusion, and analysis choices before outcome-bearing measurements are inspected. Changing any fixed choice after outcome-bearing measurements begin creates a new ADR and baseline revision; the old evidence remains attributable only to the old contract.

## Decision

### Effect and unit of analysis

The unit of analysis is one independent, matched pair of target executions. Each pair contains one observer-disabled arm and one observer-enabled arm using the same workload, input, seed, concurrency, duration, host configuration, and measurement boundaries. Enabled and disabled differ only in whether the SIA observation mechanism under evaluation is active. SIA remains otherwise identically installed and configured.

For strictly positive scale endpoints, the pair effect is `log(enabled / disabled)`. The reported estimate and interval are exponentiated ratios. For endpoints that may be zero or whose units have direct practical meaning, the pair effect is `enabled - disabled`. The reported estimate is the arithmetic mean paired difference. Distribution endpoints are reduced within each trial as declared below and then compared within pairs; individual samples within a trial are not independent replicates.

Observer effect means the paired ratio or difference between otherwise identical disabled and enabled executions. Positive and negative effects are treated symmetrically. An improvement outside a margin fails equivalence just as a regression outside the same margin does, because either can indicate uncontrolled interference or a changed workload.

### Fixed workloads

The source revision under test must contain the workload driver and input manifest referenced by the baseline. Their content digests are recorded before execution. The following four workload IDs are mandatory:

| Workload | Fixed operation | Input, concurrency, and duration | Warm-up |
|---|---|---|---|
| `cpu_fixed` | Deterministic dense floating-point matrix multiplication using the repository benchmark driver | seed 104729; matrices generated once from that seed; one target worker; 180 measured seconds | one identical 30-second unmeasured execution |
| `io_fixed` | Repeated sequential reads followed by writes of a preallocated 8 GiB benchmark file using direct I/O and 1 MiB blocks | seed 130363 for file contents; queue depth 1; one target worker; 180 measured seconds; file-system and mount options recorded | one 30-second unmeasured read/write execution; test file recreated before each pair |
| `gpu_fixed` | Deterministic accelerator matrix multiplication through the target runtime | seed 155921; fixed 8192 by 8192 operands; one host submitter and one accelerator stream; 180 measured seconds | 30 seconds plus device synchronization |
| `gui_active_fixed` | The same `gpu_fixed` target while the SIA GUI is visible at its fixed dashboard view, window size 1280 by 720, refresh interval 1000 ms, and compositor state recorded | seed 155921; one host submitter and one accelerator stream; 180 measured seconds | 30 seconds with the GUI already visible plus device synchronization |

The benchmark manifest records the exact driver revision, command arguments, generated-input digests, runtime, and dependency versions. Substitution of a synthetic workload for a workload named above is not allowed. Synthetic results may be supplementary but cannot establish a target claim.

The machine is dedicated to a benchmark block. Network services not required by the target are disabled, scheduled jobs are stopped, and no interactive work is performed. CPU affinity, NUMA policy, memory limits, accelerator selection, power mode, fan policy, and CPU/GPU frequency policy are fixed for the block and recorded. The host is rebooted before the first block unless a documented target deployment constraint makes rebooting impossible; that constraint and the common starting procedure are then recorded.

### Observer configuration and boundaries

The enabled arm uses the release configuration being claimed, including all collectors, sampling intervals, storage, GUI refresh, buffering, export, and retention behavior. The disabled arm uses the same binary, configuration, target, and lifecycle with the observation mechanism disabled. Configuration files and effective configuration are content-addressed in the baseline.

An external harness is authoritative. It starts timing immediately before target launch and stops only after the target exits, accelerator work is synchronized, SIA collectors stop, buffers and exports are flushed, and observer teardown completes. Deferred observer work therefore remains in the enabled arm. The harness uses a monotonic clock and records its clock source and resolution.

Harness calibration consists of at least 30 paired no-op executions using the same launch, wait, synchronization, and collection path. Its elapsed-time distribution, CPU time, I/O, and sampling load are recorded. Calibration overhead is not subtracted unless a subtraction rule is approved in a replacement ADR; it is used to detect an incapable harness. The harness is incapable, and the affected block is invalid, if its p95 elapsed-time cost exceeds one quarter of the smallest elapsed-time equivalence margin in absolute time or if its sampling causes observed loss or lateness gates to fail.

Accelerator start and stop timestamps bracket explicit device synchronization. Power, clock, temperature, and utilization samples use a recorded common monotonic time base or a recorded clock-offset conversion. Buffering, export, synchronization, and teardown costs are never moved beyond the enabled boundary.

### Required endpoints and margins

Every endpoint listed for a workload in the baseline is required. Ratios use reciprocal, directionally symmetric limits. Difference margins are symmetric about zero.

| Endpoint | Unit and aggregation | Equivalence margin | Practical rationale |
|---|---|---|---|
| Target elapsed p50 | ratio of per-trial median operation latency | 0.98 to 1.02 | A 2% change is the largest tolerable scheduling impact for interactive and batch use. |
| Target elapsed p95 | ratio of per-trial p95 operation latency | 0.98 to 1.02 | Tail impact beyond 2% is operationally visible. |
| Target throughput | ratio of completed operations per measured second | 0.98 to 1.02 | A sustained 2% capacity change is material. |
| SIA CPU time | difference in SIA CPU seconds divided by measured wall seconds | -0.01 to 0.01 | One percentage point of a CPU is the allowed host budget. |
| SIA peak RSS | difference in MiB | -32 to 32 MiB | 32 MiB is the allowed resident-memory budget. |
| SIA bytes read | difference in MiB per 180-second trial | -4 to 4 MiB | This bounds storage read disturbance. |
| SIA bytes written | difference in MiB per 180-second trial | -4 to 4 MiB | This bounds persistent and buffered write disturbance. |
| Target GPU clock | ratio of per-trial median active clock | 0.98 to 1.02 | A 2% clock shift can change accelerator completion time. |
| Target GPU power | ratio of per-trial mean power | 0.97 to 1.03 | A 3% power shift is the maximum immaterial energy or boost disturbance. |
| Target GPU temperature | difference in per-trial maximum degrees Celsius | -2 to 2 C | Two degrees is the largest acceptable thermal perturbation. |
| Sample-loss rate | difference in attempted samples lost divided by attempted samples | -0.001 to 0.001 | A 0.1 percentage-point arm effect protects measurement integrity. |
| Collector lateness p95 | difference in milliseconds | -5 to 5 ms | Five milliseconds is the maximum acceptable scheduling displacement. |
| SIA GPU activity | difference in percentage points of mean GPU busy time | -1 to 1 percentage point | GUI rendering must consume no more than one percentage point of device time. |

CPU and I/O workloads require elapsed p50, elapsed p95, throughput, SIA CPU time, peak RSS, bytes read, bytes written, sample-loss rate, and collector-lateness p95. GPU requires those endpoints plus GPU clock, power, and temperature. GUI-active requires all GPU endpoints plus SIA GPU activity.

In addition to paired equivalence, each enabled trial must have sample loss no greater than 0.1%, collector-lateness p95 no greater than 5 ms and collector-lateness maximum no greater than twice the configured sample interval. Each GUI-active enabled trial must contain a valid SIA GPU-activity measurement. These are measurement-quality gates, not replacements for paired equivalence.

### Randomization, blocking, and environmental control

A pair is assigned `disabled-enabled` or `enabled-disabled` by a pseudorandom permutation generated from seed 20260905. Each consecutive block of four pairs contains two of each order. Workloads are run in blocks, and workload-block order is independently permuted with the same recorded generator. The complete schedule is generated and committed to the baseline before outcome-bearing runs begin.

Both periods of a pair use the same input identity. A 10-minute cool-down separates periods and continues until CPU package temperature and accelerator temperature are each within 2 C of that pair's first-period pre-run value for five consecutive one-minute observations. Failure to recover within 30 minutes invalidates the pair for thermal carryover and is recorded without replacement unless the preregistered maximum permits another newly scheduled pair.

Warm-up is performed separately for each period and is excluded from the measured window. For I/O, cache state is made identical by direct I/O, recreation of the fixed test file, and a recorded sync procedure. For CPU and accelerator workloads, the declared warm-up brings code, allocations, and runtime compilation to the same state. Thermal or power throttling flags, frequency-policy changes, kernel warnings, target errors, unexpected processes consuming more than 1% CPU, and background I/O above 10 MiB during a period are recorded.

Analysis includes the paired arm effect as the estimand and reports order and period diagnostics. Before arm labels are used for equivalence, the analyst tests an order-by-arm term and a period term in a fixed-effects model containing pair as a block. A family-adjusted diagnostic p-value below 0.05, or an estimated order/carryover effect larger than half the endpoint margin, makes that workload-endpoint inconclusive. It is not adjusted away after inspection. Cache-state mismatch, throttling in only one arm, failed thermal recovery, or material background-interference threshold breach invalidates the pair under the objective rules below.

### Repetitions and stopping

The confidence target is at least 95% simultaneous family-wise coverage. The precision target is an adjusted confidence-interval half-width no greater than one half of the endpoint's equivalence half-margin.

Each workload begins with 10 independently randomized pilot pairs. For every required endpoint, compute the sample standard deviation `s` of its pair effects on the analysis scale. With `K` equal to the number of required workload-endpoint conclusions in the committed baseline, let `z` be the standard-normal quantile `1 - 0.05/(2K)`. The required count for an endpoint is `ceil((z * s / (margin_half_width / 2))^2)`. The fixed final count for a workload is the maximum endpoint count for that workload, bounded below by 30 independent valid pairs and above by 200.

If the calculated count exceeds 200, that workload is declared inconclusive; the margin or maximum cannot be changed using those outcomes. Pilot observations enter the final analysis only if they used the committed schedule, configuration, harness, inputs, warm-up, and eligibility rules. Otherwise they are calibration-only and new pilot pairs are collected. Once the final count is fixed, there is no early success or futility stop and no sample-size re-estimation. If exclusions leave fewer than the fixed count, new schedule entries may be executed only until 200 attempts; otherwise the affected results are inconclusive. This fixed rule and simultaneous intervals control optional stopping and type-I error.

### Confidence intervals and multiplicity

For each endpoint, compute the mean paired effect and a two-sided Student t confidence interval at confidence level `1 - 0.05/K`, where `K` is fixed from the complete required matrix before execution. Pair effects must be independent across pairs and approximately normal. Ratio endpoints are analyzed on the log scale and exponentiated only for reporting. Difference endpoints remain in their declared units.

Bonferroni construction supplies at least 95% family-wise coverage across all required conclusions, including workloads. No conclusion is removed from `K` because it is missing, failed, or inconvenient. If normality is materially contradicted by a preregistered Shapiro-Wilk check at family-adjusted 0.01 or by an absolute standardized skewness above 1, use a two-sided percentile bootstrap interval with 100,000 deterministic resamples and Bonferroni confidence `1 - 0.05/K`. Resampling is by independent pair within workload and preserves the full vector of that pair's endpoints. Bootstrap seed 32452843 is fixed. Too few complete pairs for the selected method is inconclusive.

Equivalence is established only when the complete adjusted interval is strictly inside the predeclared lower and upper margin. Touching or overlapping a margin fails. A favorable estimate, an unadjusted interval, or failure to reject a difference null hypothesis is insufficient.

Overall low observer effect passes only if every required workload-endpoint result passes equivalence and every quality and validity gate passes. A failed, missing, invalid, or inconclusive endpoint makes the overall result non-passing. Failures and conflicts must be reported alongside successes; no subset or aggregate may be presented as the overall conclusion.

### Attempt accounting and exclusions

Every attempted period is appended to the raw observations or a losslessly reproducible, content-addressed raw reference before inspecting its arm comparison. Records include workload, endpoint samples, pair, order, period, timestamps, arm, units, inclusion status, and reason.

Objective pair exclusions are limited to: target or harness non-zero exit; timeout at 240 seconds; host reboot or suspend; loss of the external timing record; input-digest mismatch; enabled/disabled configuration mismatch beyond observer mode; thermal recovery failure; kernel-reported device reset; arm-specific throttling; unexpected process above 1% CPU for more than 10% of the measured period; background I/O above 10 MiB during a period; or missing required measurement-quality evidence. An exclusion applies to the entire pair and all its endpoints. Criteria are evaluated without comparing arm outcomes. Statistical outliers are retained. No observation is winsorized or removed because of its value.

Timeouts and target failures retain their elapsed boundary and failure class but do not receive invented throughput or resource values. A required endpoint affected by censoring is inconclusive unless both arms provide the complete preregistered endpoint. Missing samples contribute to attempted and lost counts. Collector failures, missing GUI GPU activity, absent units, clock discontinuities, or an unavailable provenance field make affected workload-endpoint results inconclusive and prevent an overall pass.

The baseline reports attempted periods, attempted pairs, valid complete pairs, excluded pairs by reason, timeouts, failures, censored values, and missing values. Replacement attempts retain new pair identifiers and do not erase excluded attempts.

### Baseline and provenance

The separate YAML artifact is both the preregistration record and the result envelope. It defines the required matrix, margins, raw-observation schema, provenance schema, and deterministic result fields. When measurements exist, it contains arm-labeled paired raw data or content-addressed references that reproduce it losslessly; summaries alone are insufficient.

Required provenance includes source and ADR revisions, effective configuration and observer mode, host identity, OS, kernel, CPU, memory, accelerator and driver/runtime versions, firmware where relevant, power and frequency settings, isolation controls, workload/input identity and digests, harness and measurement-tool versions, clock source, and schedule revision. Development-host and target-hardware evidence are labeled separately.

A target low-observer-effect claim requires measurements on the declared target hardware and deployment-representative software configuration. Development-host, unmatched-host, synthetic-only, incomplete, schema-only, or uncalibrated evidence cannot support it.

### Claim rule

This proposed ADR and its unexecuted baseline do not claim low observer effect. A claim is permitted only after this procedure is approved and run unchanged on the declared target, all provenance and raw evidence are reviewable, an independent implementation reproduces the results, and every required adjusted interval and quality gate passes. Until then the deterministic overall status is `inconclusive`.

## Consequences

The contract favors falsifiability over a small benchmark. It can reject or withhold a low-observer-effect claim because of real overhead, inadequate precision, environmental contamination, measurement-quality failure, incomplete evidence, or an unmatched target. Such outcomes are intentional and may not be selectively summarized away.