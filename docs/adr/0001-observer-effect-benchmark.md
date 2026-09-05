# ADR 0001: Observer-effect benchmark contract

- Status: Proposed
- Specification revision: 4479bcc19db72f6ad243a87e4b7271496d60d0b7
- Requirements: R01, R12, R38, R40
- Baseline artifact: benchmarks/observer-effect-baseline.yaml

## Decision

SIA low-observer-effect claims require a preregistered paired equivalence experiment on the declared target hardware. The external benchmark harness is authoritative. An overall pass is permitted only when every required workload-endpoint combination establishes equivalence and every measurement-quality gate passes.

This proposed ADR and its unexecuted baseline schema are not evidence of low observer effect. Until the procedure is completed on target hardware, the only valid overall conclusion is inconclusive.

## Requirement traceability

| Requirement | Contract provision |
| --- | --- |
| R01 | Observer effect is measured by paired observer-disabled and observer-enabled runs with fixed workloads, external timing, raw evidence, and simultaneous equivalence conclusions. |
| R12 | Per-trial SIA CPU time, RSS, I/O, sample loss, collector lateness, and GUI GPU activity are recorded. |
| R38 | Accelerator trials record target GPU clock, power, temperature, synchronization, and SIA GPU activity. |
| R40 | Provenance, measurement boundaries, quality gates, deterministic analysis, and target-hardware claim restrictions are fixed here and in the baseline artifact. |

## Claim and unit of analysis

One independent matched pair is the unit of analysis. A pair consists of one observer-disabled trial and one observer-enabled trial using the same workload, input, seed, concurrency, duration, host configuration, and measurement boundaries.

For ratio endpoints, the paired effect is `log(enabled / disabled)`. Reported estimates and confidence bounds are exponentiated back to ratios. For difference endpoints, the paired effect is `enabled - disabled` in the endpoint unit. Every margin is directionally symmetric: ratio margins are symmetric on the log scale and difference margins are `[-M, +M]`.

A favorable point estimate, overlap with a margin, failure to reject a difference test, or only part of a confidence interval lying within a margin does not establish equivalence. Equivalence requires the complete multiplicity-adjusted confidence interval to lie strictly inside both bounds of the predeclared margin.

## Fixed workloads

Each trial has a 60-second stabilization period, a 30-second workload warm-up, and a 300-second measured interval. One target workload process is used. No other user workload may run. Exact executable versions are captured before the first pilot pair and may not change within an experiment.

| ID | Definition | Input and concurrency | Authoritative target endpoints |
| --- | --- | --- | --- |
| cpu_batch | `stress-ng --cpu 1 --cpu-method matrixprod --timeout 330s --metrics-brief` | One worker; first 30 seconds warm-up; fixed stress-ng build recorded in provenance | elapsed time and matrix-product bogo operations per second |
| storage_batch | `fio --name=sia-seq --rw=readwrite --rwmixread=50 --bs=1M --size=8G --iodepth=1 --numjobs=1 --direct=1 --time_based=1 --runtime=330 --randseed=4479` | One job in an otherwise empty dedicated filesystem; the file is recreated before each pair | elapsed time and aggregate read-plus-write MiB per second |
| gui_active | `glmark2 --fullscreen --run-forever` under the recorded native display session | One instance; default scene sequence; fixed glmark2 build and display geometry recorded in provenance | elapsed time and frames per second |

The harness terminates each workload after exactly 330 seconds and analyzes only the final 300 seconds. A workload version, command, input, display geometry, filesystem, concurrency, duration, or seed change starts a new experiment and baseline.

## Required endpoints and margins

All workloads require the first six endpoints below. Accelerator endpoints additionally apply to `gui_active`, giving 22 required conclusions in total.

| Endpoint | Unit | Aggregation within a trial | Effect | Equivalence margin | Practical rationale |
| --- | --- | --- | --- | --- | --- |
| target_elapsed_time | ms | External monotonic time from measured-work release to synchronized target completion | ratio | 0.98 to 1.02 | A two-percent timing change is the largest operationally negligible delay. |
| target_throughput | workload unit/s | Total completed work divided by the measured 300 seconds | ratio | 0.98 to 1.02 | A two-percent throughput change is the largest operationally negligible loss or gain. |
| sia_cpu_time | ms | Process-tree user plus system CPU time during the measurement boundary | difference | -3000 to 3000 | Three seconds is one percent of one CPU core over the trial. |
| sia_peak_rss | bytes | Maximum process-tree RSS during the measured interval | difference | -209715200 to 209715200 | 200 MiB is the maximum negligible memory displacement. |
| sia_read_bytes | bytes | Process-tree storage bytes read during the measured interval | difference | -16777216 to 16777216 | 16 MiB avoids materially perturbing storage and cache behavior. |
| sia_write_bytes | bytes | Process-tree storage bytes written, including deferred export, through teardown | difference | -16777216 to 16777216 | 16 MiB bounds persistent and cache-writing interference. |
| target_gpu_clock | MHz | Time-weighted mean over valid external samples | ratio | 0.98 to 1.02 | A two-percent clock shift is the largest negligible accelerator perturbation. |
| target_gpu_power | W | Time-weighted mean over valid external samples | ratio | 0.98 to 1.02 | A two-percent power shift is the largest negligible energy and thermal perturbation. |
| target_gpu_temperature | degC | Maximum during the measured interval | difference | -2 to 2 | Two degrees Celsius is the largest negligible thermal shift. |
| sia_gpu_activity | percentage points | Time-weighted mean SIA-attributed GPU busy percentage | difference | -1 to 1 | One percentage point bounds GUI observer GPU demand. |

Elapsed time and throughput are both retained because duration control alone can hide incomplete work. CPU time, RSS, and I/O are SIA process-tree measurements in both arms; observer-disabled means SIA runs with the observation mechanism disabled, not that the SIA process is absent.

## Measurement-quality gates

Every trial records attempted and received collector samples, sample-loss count and rate, collector lateness maximum and p99, configured sampling interval, and collector clock source. Every GUI-active trial also records SIA-attributed GPU activity.

A trial passes quality gates only when sample loss is at most 1 percent, p99 collector lateness is at most one configured sampling interval, clocks remain monotonic, and every required quality field is present. Missing quality evidence makes every affected workload-endpoint conclusion inconclusive. It cannot be imputed or treated as zero.

## Pairing, order, and environment control

Pairs use identical inputs and an unchanged host. Observer order is determined before collection using alternating randomized blocks of four: two enabled-first and two disabled-first pairs, shuffled with the experiment seed recorded in the baseline. Replacement pairs retain the failed pair's assigned order.

The machine is isolated from interactive use, scheduled jobs, updates, indexing, and unrelated network traffic. CPU and GPU governors, power limits, frequency settings, affinity, display configuration, storage mount, ambient controls, and relevant environment variables are fixed and recorded. Both arms use the same SIA build and configuration and differ only by the observer-enabled switch.

The following controls are mandatory:

- Order and period are included as fixed terms in the final paired linear model. A workload is inconclusive if an order-by-arm interaction adjusted p-value is below 0.05 or its estimated magnitude exceeds half the endpoint margin.
- Warm-up is excluded by the external harness. Warm-up counters are retained as diagnostic evidence.
- Filesystem and page-cache state follow the same recorded reset procedure before both arms. Failure to complete the reset invalidates the pair before either outcome is inspected.
- A thermal reset requires target CPU and GPU temperatures to return within 2 degC of the pair's recorded precondition and no throttling flag for 60 seconds. Failure within 20 minutes invalidates the pair.
- Any firmware, kernel, driver, runtime, governor, power, frequency, display, or observer configuration change invalidates the experiment.
- Background CPU above 2 percent of one core, background storage above 1 MiB/s, an unrelated GPU client, frequency throttling, thermal throttling, OOM activity, suspend, clock discontinuity, or harness loss invalidates the pair using external logs.
- Carryover is tested from pretrial temperature, cache-reset evidence, warm-up throughput, and preceding arm. A preceding-arm effect larger than half a margin or with adjusted p below 0.05 makes the workload inconclusive; it is not repaired by selecting later pairs.

## Repetitions and stopping

The confidence target is at least 95 percent family-wise coverage across all 22 required conclusions. The precision target is a simultaneous confidence-interval half-width no greater than half of the corresponding equivalence half-margin.

Collect 20 pilot pairs per workload, followed by enough pairs to reach a minimum of 30 and no more than 200 pairs per workload. Pilot observations enter the final analysis and must follow the same contract.

After exactly 20 valid pilot pairs, calculate the required count independently for every endpoint as:

`ceil((z(1 - 0.05 / (2 * 22)) * pilot_sd / precision_target)^2)`

For ratio endpoints, `pilot_sd` and `precision_target` are on the log scale. The workload target count is the maximum of 30 and every endpoint count, capped at 200. This count is fixed once and recorded before further observations. There is no interim equivalence testing or sample-size recalculation. If a computed count exceeds 200 or the precision target is not met at 200, the affected result and overall result are inconclusive. This single blinded variance-based re-estimation and prohibition on outcome-driven stopping prevent optional-stopping inflation.

## Confidence intervals and multiplicity

For each endpoint, fit the paired effect with order and period terms and report the adjusted arm-effect estimate. Construct a two-sided `1 - 0.05 / 22`, or approximately 99.7727 percent, confidence interval using a pair-cluster bootstrap with 10,000 resamples. The entire pair is the resampling unit. Resampling uses the SHA-256-derived integer seed recorded in the baseline and preserves workload strata. Ratio endpoints are bootstrapped on the log scale and exponentiated after interval construction.

Bonferroni adjustment across the 22 preregistered conclusions provides at least 95 percent family-wise coverage without relying on endpoint independence. No endpoint may be removed from the family after data collection. If the model or bootstrap cannot produce finite bounds, the result is inconclusive.

## Failures, missing data, censoring, and exclusions

The harness assigns an attempt ID before launch and records every attempted run, including setup failures. Allowed exclusion reasons are limited to the objective predeclared environment invalidations above, harness failure, external power interruption, workload launch failure before the measurement boundary, or operator safety stop. Exclusion decisions use external logs and are made without inspecting arm outcomes.

A target crash, timeout after measurement begins, observer crash, observer-caused resource exhaustion, incomplete observer export, or missing required measurement is not excluded as an anomaly. It is recorded as a failed or censored observation and makes the affected endpoint inconclusive. Timeouts are right-censored at the fixed timeout; they are never substituted with the timeout value for equivalence analysis. Unexpected values remain included unless an allowed cause is independently documented. Both members of an invalid pair remain in the attempt ledger, and neither enters quantitative analysis.

The baseline reports attempted runs, included pairs, exclusions by reason, failures, timeouts, censored values, and missing values. Exclusions cannot be selected after comparing arms.

## Authoritative boundaries and observer work

The external harness clock begins immediately before releasing measured workload activity. It ends only after target accelerator work is synchronized, SIA buffers are flushed, exports complete, and observer teardown completes. Enabled-arm startup, sampling, synchronization, buffering, serialization, export, flush, and teardown costs are included. Deferred work may not be moved outside the enabled boundary.

The harness records its own CPU, RSS, I/O, timing, and sampling cost in both arms. Its version and configuration are identical between arms. GPU work is explicitly synchronized before timestamps and counters are finalized. The observer and harness use a recorded monotonic clock mapping; failed synchronization is an invalidation, not an estimate.

## Evidence and conclusion rules

The baseline contains provenance, an attempt ledger, arm-labeled paired raw observations or immutable content-addressed references, quality evidence, and one deterministic result row for every required workload-endpoint combination. Summaries alone are insufficient.

A workload-endpoint passes only if its required fields and quality evidence are complete, its sample target and precision target are met, no invalidating order or carryover result exists, and its complete adjusted confidence interval lies strictly inside its margin. Otherwise it is failed when a complete interval establishes nonequivalence, or inconclusive when evidence is missing, invalid, underpowered, censored, or non-finite.

The overall conclusion passes only when all 22 results pass. Any failed result makes the overall conclusion fail. Any inconclusive or missing result makes it inconclusive unless another result already makes it fail. Conflicting outcomes must be reported together and may not be selectively summarized.

Development-host evidence is labeled development only. It cannot support a target claim. Target low-observer-effect claims are forbidden when hardware is unmatched, workloads are synthetic-only relative to the claimed production use, endpoint coverage is incomplete, estimates are uncalibrated, or provenance and quality gates are incomplete.