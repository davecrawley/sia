# ADR 0001: Observer-effect benchmark contract

- Status: Proposed
- Specification revision: `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/observer-effect-baseline.json`

## Decision

Low observer effect is a statistical equivalence claim made only from paired runs on the declared target hardware. An observer-disabled arm and an observer-enabled arm use the same SIA revision, workload, input, seed, host, configuration, and measurement boundary. They differ only in whether the observation mechanism under evaluation collects, buffers, exports, and tears down its data.

This proposed ADR and its empty baseline are a preregistration, not evidence. They do not establish low observer effect. A claim is permitted only after the fixed procedure below has run on the declared target and every required result and quality gate passes.

## Unit of analysis and effects

The independent unit is a completed pair containing one disabled and one enabled arm. Hosts or processes reused within a pair do not create additional independent observations.

For strictly positive performance endpoints, the pair effect is `log(enabled / disabled)`. Results are reported after exponentiation as an enabled/disabled ratio. For additive resource, accelerator, and quality endpoints, the pair effect is `enabled - disabled` in the endpoint's stated unit. All equivalence margins are directionally symmetric: log-ratio limits are `[-log(M), +log(M)]`, reported as `[1/M, M]`; difference limits are `[-D, +D]`.

The baseline artifact fixes the required workloads, endpoints, units, arm aggregation, effect type, margin, and rationale. Each margin represents the largest change considered operationally negligible for the benchmark's stated scope. A favorable direction receives no larger allowance than an unfavorable direction.

## Required workloads

The suite contains four fixed workloads:

1. `cpu_fixed`: one-thread sysbench CPU work, prime limit 20000, deterministic seed 104729, 60 seconds warm-up and 180 seconds measured duration.
2. `io_fixed`: one-job fio mixed sequential read/write workload on an isolated 8 GiB file, 1 MiB blocks, synchronous direct I/O, deterministic seed 104729, 60 seconds warm-up and 180 seconds measured duration.
3. `gpu_offscreen_fixed`: off-screen glmark2 at 1920x1080 using the listed scenes in the listed order, 60 seconds warm-up and 180 seconds measured duration.
4. `gui_active_fixed`: on-screen glmark2 at 1920x1080 using the same fixed scene sequence, with the SIA GUI visible and unchanged throughout, 60 seconds warm-up and 180 seconds measured duration.

Executable paths, exact tool versions, resolved arguments, input identities, and hashes are mandatory provenance fields. A different tool, scene, input, concurrency, resolution, duration, or seed is a different benchmark and requires a new proposed ADR revision before collection. These controls characterize the declared suite only. A target-production claim additionally requires a target-representative nonsynthetic workload preregistered by ADR amendment; synthetic-only evidence may not be generalized to production workloads.

## Endpoints and aggregation

The external harness is authoritative for elapsed time, throughput, process CPU time, RSS, process I/O, accelerator telemetry, and measurement boundaries. Raw samples and per-arm aggregates are retained.

The primary workload endpoint is throughput for the time-based CPU, I/O, and graphics workloads. Throughput is total completed work divided by the 180-second measurement window. SIA CPU time is user plus system CPU milliseconds divided by measured target seconds. RSS is the time-weighted mean MiB from external process-tree samples. I/O is the process-tree byte-counter delta divided by measured target seconds. GPU clock, board power, temperature, and utilization are time-weighted means over timestamped external samples. Sample loss is missing sequence numbers divided by expected sequence numbers. Collector lateness is the empirical p99 of `collector_receive_monotonic_ns - scheduled_monotonic_ns`.

Every trial records sample-loss count, expected-sample count, sample-loss rate, collector-lateness samples and p99. Every GUI-active trial also records SIA process-tree GPU utilization. Missing measurement-quality fields make every affected workload-endpoint conclusion inconclusive. Accelerator endpoints are required for both GPU workloads. The artifact's applicability matrix is exhaustive; an endpoint cannot be dropped after outcomes are seen.

Overall low observer effect passes only if every required workload-endpoint result passes. A failed, missing, invalid, or inconclusive result prevents an overall pass. Conflicting endpoints must be reported together and cannot be selectively summarized.

## Equivalence and uncertainty

Let `K` be the number of required workload-endpoint conclusions in the frozen applicability matrix. For each conclusion, compute the mean of its independent pair effects and a two-sided paired Student-t confidence interval with confidence `1 - 0.05/K`, using critical value `t(1 - 0.05/(2K), n-1)`. Bonferroni therefore supplies at least 95% family-wise coverage across all required conclusions without relying on endpoint independence. Log-ratio intervals are exponentiated for reporting.

The paired-t procedure assumes independent pairs and a finite-variance, approximately normal distribution of pair effects. Raw effects, a Q-Q diagnostic, and skewness are reported. Absolute skewness above 2 or a visually material Q-Q departure recorded by the predeclared automated diagnostic invalidates that conclusion; it is inconclusive and requires a new preregistered robust design, not a post-hoc method change.

Equivalence is established only if the complete adjusted confidence interval lies strictly inside both symmetric margin limits and all data-quality gates pass. Touching or overlapping a margin, failure to reject a difference test, an unadjusted interval, or a favorable point estimate is insufficient.

## Repetition and stopping

A separate pilot of 12 independent pairs per workload estimates the standard deviation `s` for each required pair effect. Pilot observations never enter confirmatory analysis. For each endpoint, calculate

`n = ceil(((z(1-0.05/(2K)) + z(0.90)) * s / (margin/2))^2)`.

For log-ratio endpoints the margin in this formula is `log(M)`. The workload's confirmatory count is the maximum endpoint-specific `n`, bounded below by 30 pairs and above by 200 pairs. The numerical precision target is an adjusted 95%-family-wise half-width no greater than half the equivalence margin, with 90% planning power at that target. If the formula exceeds 200, collection does not begin and the result is inconclusive for insufficient attainable precision. If the realized adjusted half-width exceeds the target at the fixed count, the result is inconclusive; no extra pairs are added. There is no outcome-dependent stopping, peeking, sample-size revision, or reuse of pilot data.

## Pairing, order, and environmental control

Each pair uses matched workload inputs, seeds, concurrency, duration, host image, boot, power settings, frequency settings, and observer configuration. Pair order is assigned before collection by sorting `SHA-256(experiment_id || workload_id || pair_id)` on its low bit: zero is disabled-first and one is enabled-first. Equal allocation is enforced within each workload by deterministically flipping the final excess assignments. Analysis includes the within-pair effect and reports order and period strata.

Before each arm, the harness restores the workload fixture, clears only the benchmark-owned cache when the workload definition requires it, waits for the fixed 60-second thermal idle condition, performs the fixed warm-up, and verifies isolation. Cache state is never changed between arms selectively. The host is dedicated: no interactive users, scheduled jobs, updates, unrelated containers, or dynamic power policy.

The harness records arm order, period, cache-reset result, warm-up result, ambient and device temperatures, clock and power states, throttling flags, background CPU/I/O, and carryover probes. A pair is invalidated by a failed reset; warm-up deviation over one second; thermal baseline difference over 2 degrees C; any throttling; background CPU above 2% for more than five seconds; unrelated I/O above 10 MiB/s for more than five seconds; workload/input mismatch; or carryover probe differing by more than 3% between pre-arm checks. Criteria are evaluated from arm-blind infrastructure fields. Invalidated attempts remain in the raw ledger and are replaced using the same preassigned order until the fixed valid-pair count or attempt ceiling of twice that count is reached. Reaching the ceiling is inconclusive.

Order or period interaction is tested at family-wise alpha 0.05/K. An interaction estimate larger than half the equivalence margin, a significant interaction, or an imbalance after invalidation makes the conclusion inconclusive. No post-hoc stratification rescues a failed conclusion.

## Failures, timeouts, missingness, and anomalies

Every attempted arm receives a ledger row. Timeout is fixed at measured duration plus 30 seconds. Crashes, timeouts, absent endpoints, corrupt timestamps, failed synchronization, censoring, or unmatched arms invalidate the pair and are reported with an objective reason code. Numerical outliers are retained. Exclusion is allowed only for the predeclared infrastructure criteria above and is decided without inspecting enabled-versus-disabled endpoint outcomes. Counts of attempts, valid pairs, exclusions, failures, timeouts, missing values, and censored values are reported for every result.

Any arm failure-rate difference greater than 2 percentage points or any missingness associated with arm, order, or endpoint makes the affected result inconclusive. Censored values are not imputed.

## Measurement boundary and harness cost

Setup ends only after isolation checks and warm-up. The arm begins immediately before the target is released from a synchronization barrier. CPU and accelerator work is synchronized at both boundaries; GPU queues are fenced before the end timestamp. The enabled arm continues through buffer drain, export flush, and observer teardown until the observer and its children exit and external counters are read. Deferred observer work is never omitted or moved outside the enabled arm.

Before pilot collection, the external harness runs 20 harness-only calibration repetitions in each mode. Its median timing cost must be below 10% of the smallest time margin, sampling CPU below 10% of the SIA CPU margin, and sampling I/O below 10% of the I/O margin. Calibration, clock source, resolution, synchronization error, sample cadence, buffering behavior, and tool versions are retained. Failure is inconclusive until the harness is improved and this ADR is revised.

## Provenance and claim scope

The dataset records source and ADR revisions, observer mode and configuration hash, host OS and kernel, CPU, memory, accelerator, driver and runtime versions, firmware, power and frequency settings, environment controls, workload/input identity, tool versions, timestamps, and clock source. Required provenance may not be null in an executed dataset.

Development-host evidence is labeled development-only. Target-hardware evidence requires exact declared target identity. Unmatched hosts, uncalibrated estimates, incomplete endpoints, incomplete quality fields, synthetic-only workloads, or nonreproducible summaries cannot support a target low-observer-effect claim.

## Requirement traceability

| Requirement | Contract decision |
| --- | --- |
| R01 | Paired enabled/disabled effects, fixed symmetric margins, exhaustive equivalence gate |
| R12 | External authoritative timing, complete resource and accelerator endpoints, deferred-work boundary |
| R38 | Raw pair ledger, measurement-quality evidence, provenance, deterministic conclusions |
| R40 | Proposed ADR and baseline; no claim until target execution satisfies every gate |