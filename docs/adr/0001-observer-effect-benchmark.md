# ADR-0001: Observer-effect benchmark contract

- Status: Proposed
- Specification: `spec/SPECIFICATION.md` at `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/observer-effect-baseline.yaml`

## Context

SIA must not claim low observer effect from an uncalibrated estimate, an unmatched development host, synthetic-only evidence, incomplete endpoint coverage, a favorable point estimate, or the existence of this unexecuted plan. This ADR preregisters the experiment and analysis required before such a claim is permitted.

## Decision

### Effect and unit of analysis

The unit of analysis is one independent matched pair: one observer-disabled run and one observer-enabled run of the same workload, input, seed, host configuration, and measurement window. The external harness, not SIA, supplies authoritative timing and resource measurements.

For positive ratio endpoints, pair `i` has effect `d_i = log(enabled_i / disabled_i)`. For additive endpoints, `d_i = enabled_i - disabled_i`. The reported ratio estimate is `exp(mean(d_i))`; additive estimates use `mean(d_i)`. Observer effect is therefore the paired difference or ratio between otherwise identical arms. Enabled means the complete observation path, including collection, buffering, export, flush, synchronization, and teardown, is active. Disabled retains the same target and harness but does not start that observation path.

All margins are directionally symmetric. A ratio margin of `m` means the interval must lie wholly inside `[-log(m), +log(m)]`, equivalent to ratios `(1/m, m)`. An additive margin is `[-m, +m]`.

### Fixed workload set

Every target-hardware evaluation must run all four workloads below. Tools and their versions or immutable image digests are recorded in the baseline before collection.

| ID | Definition | Input, concurrency, and boundary |
|---|---|---|
| `cpu_fixed` | Fixed-work CPU hashing | SHA-256 a 1 GiB deterministic byte stream generated from seed `0x53494101`, 32 complete passes, one target worker. Measure from immediately before pass 1 through completion of pass 32. |
| `io_fixed` | Fixed-work sequential file read | Read a preallocated 16 GiB file whose bytes are generated from seed `0x53494102`, using direct I/O, 1 MiB blocks, queue depth 1, one target worker, four complete passes. Measure all four passes. The file is created before warm-up and its digest is recorded. |
| `gpu_fixed` | Fixed-work accelerator compute | Execute 10,000 iterations of a deterministic 4096-by-4096 FP32 matrix multiply using seed `0x53494103`, batch size 1, one host submitter, explicit device synchronization at the end. Measure from before the first submission through final synchronization. |
| `gui_active` | Fixed GUI and accelerator presentation | Run a 1920x1080 off-screen scene containing 10,000 deterministic sprites from seed `0x53494104` for 120 seconds at uncapped presentation, one GUI process; SIA's GUI is visible and continuously refreshing at its declared production interval in the enabled arm. Measure from the first post-warm-up frame through final presentation synchronization. |

Each arm receives a 30-second unmeasured warm-up using the same workload and input. Warm-up output is discarded. A 120-second idle cooldown separates arms and pairs. The CPU and I/O workloads must run for at least 30 seconds; if the fixed definition completes sooner on the target, its pass count is multiplied by the smallest integer that reaches 30 seconds, and that multiplier is fixed from the pilot for all subsequent runs. No other workload, seed, concurrency, duration, observer setting, or boundary may change after the pilot.

The host is dedicated to the experiment. Networking, scheduled work, automatic updates, dynamic screen savers, and unrelated user processes are disabled. CPU affinity, governor, power limits, accelerator clocks/power policy, display mode, storage device, filesystem/mount options, ambient controls, and relevant environment variables are fixed and recorded. Enabled and disabled arms differ only by the observation mechanism under evaluation.

### Required endpoints and equivalence margins

The following core endpoints are required for every workload. Accelerator endpoints are additionally required for `gpu_fixed` and `gui_active`; `sia_gpu_busy_pct` is additionally required for `gui_active`.

| Endpoint | Observation unit | Pair effect and aggregate | Equivalence margin | Practical rationale |
|---|---|---|---|---|
| `target_elapsed_ns` | ns | log ratio; geometric mean ratio | 1.03 | A change beyond 3% is operationally visible in fixed work. |
| `target_throughput_units_per_s` | workload units/s | log ratio; geometric mean ratio | 1.03 | A change beyond 3% materially changes delivered work. |
| `sia_cpu_time_ns` | ns | enabled-minus-disabled, normalized as percentage points of wall time | 2 percentage points | More than 2% of one core is material observer work. |
| `sia_peak_rss_bytes` | bytes | enabled-minus-disabled; arithmetic mean | 134217728 bytes | 128 MiB is material resident-memory pressure. |
| `sia_io_read_bytes` | bytes | enabled-minus-disabled; arithmetic mean | 16777216 bytes | More than 16 MiB per trial can perturb storage workloads. |
| `sia_io_write_bytes` | bytes | enabled-minus-disabled; arithmetic mean | 16777216 bytes | More than 16 MiB per trial indicates material deferred/export work. |
| `target_gpu_clock_hz` | Hz, time-weighted mean | log ratio; geometric mean ratio | 1.03 | A 3% clock shift can change accelerator execution. |
| `target_gpu_power_w` | W, time-weighted mean | log ratio; geometric mean ratio | 1.05 | A 5% power shift is operationally and thermally material. |
| `target_gpu_temperature_c` | degrees C, maximum | enabled-minus-disabled; arithmetic mean | 3 degrees C | A 3 C shift can alter thermal control. |
| `sia_gpu_busy_pct` | percentage points, time-weighted mean | enabled-minus-disabled; arithmetic mean | 2 percentage points | GUI observer work above 2 points materially competes for the GPU. |

Elapsed time includes process startup needed by the target, all submitted target work, synchronization of asynchronous accelerator work, and target teardown. The enabled boundary also includes observer startup before target timing and observer drain, export, synchronization, and teardown before the harness ends resource accounting. Throughput is completed deterministic units divided by external elapsed time. CPU time is user plus system time. RSS is the maximum resident set over the boundary. I/O is process-accounted physical read/write bytes where available, with the exact counter source recorded. GPU samples use device telemetry and are integrated over monotonic time; clocks and power are time-weighted means and temperature is the maximum.

A required endpoint that is unsupported on relevant target hardware is inconclusive, not waived. GPU endpoints are not required only when the declared target has no accelerator and neither the target nor SIA uses one; that fact must be recorded. Because `gpu_fixed` and `gui_active` are required for an accelerator claim, a target accelerator claim cannot use that exception.

### Measurement-quality gates

Every attempted arm records expected and received sample counts, lost-sample count and rate, collector lateness distribution and maximum, collector errors, source and monotonic timestamps, sequence numbers, export completion, and teardown completion. Every `gui_active` arm records SIA GPU activity, including a valid zero.

A pair is quality-valid only if both arms have sample loss at most 1%, p99 collector lateness at most two configured sampling periods, no clock regression, no unaccounted collector error, and complete required fields. Missing quality evidence makes every affected workload-endpoint result inconclusive. Quality gates are not evidence of equivalence.

### Pairing, order, and nuisance effects

Pair order is generated before execution from seed `0x53494105` in balanced blocks of four: two enabled-first and two disabled-first, randomly permuted. Inputs, affinity, environment, and settings are identical within a pair. Pair IDs and the complete schedule are committed before outcome inspection.

The pilot and final analysis record arm order, period, warm-up completion, cache state, CPU/GPU temperature, clocks, throttling flags, background CPU/I/O, and elapsed cooldown. The final model includes centered period and order indicators as prespecified covariates while the endpoint estimate remains the adjusted enabled-minus-disabled contrast. Results are also stratified by order as a diagnostic.

An experiment is invalidated and rerun from a newly registered schedule if there is thermal or power throttling, failed warm-up, uncontrolled frequency/power-policy change, background utilization above 5% of one CPU or storage busy time above 5% during an arm, clock regression, or evidence of carryover: an order-by-arm interaction whose family-wise adjusted interval excludes zero and exceeds half the endpoint margin. Cache-sensitive I/O uses direct I/O; inability to honor it invalidates `io_fixed`. Invalidations are applied from telemetry without consulting favorable or unfavorable arm differences.

### Repetitions and uncertainty

A separate 12-pair pilot is collected for every workload. Pilot pairs never enter the confirmatory analysis. From included pilot paired effects, let `s` be the sample standard deviation. For each required conclusion `j`, compute

`n_j = ceil((z_(1-alpha/(2K)) * s_j / h_j)^2)`

where `K` is the number of required workload-endpoint conclusions after only the predeclared no-accelerator applicability rule, `alpha = 0.05`, and `h_j` is one third of that endpoint's equivalence half-margin. The final count per workload is the maximum `n_j` over its endpoints, bounded to `[30, 200]` independent pairs. Thus the numerical precision target is an adjusted 95% family-wise interval half-width no greater than one third of the equivalence margin. Zero pilot variance uses 30 pairs. A non-finite variance or a calculated requirement above 200 makes the affected result inconclusive unless a replacement protocol is proposed and approved before new observations.

The fixed final count is collected in full. There is no efficacy, futility, or precision stopping and no optional addition or removal of observations after outcome inspection.

For the confirmatory analysis, resample whole independent pairs within workload, preserving both arms and all endpoint correlations. Use a deterministic, seeded, 100,000-replicate paired bootstrap and the two-sided percentile interval at confidence `1-alpha/K` for each mean paired effect. The baseline records `K`, seed, quantile indices, and software version. This Bonferroni construction provides at least 95% family-wise coverage without assuming endpoint independence. It assumes independent pairs and exchangeability of included pairs within workload; material violation is inconclusive. Prespecified period/order adjustment is performed within each bootstrap replicate.

Equivalence is established only when the complete adjusted confidence interval lies strictly inside both predeclared bounds. Margin overlap, failure to reject a difference test, an unadjusted interval, or a favorable point estimate is insufficient.

### Failures, exclusions, and accounting

Every attempted run receives a pair and arm record. Timeout is 2 times the pilot median arm duration or 10 minutes, whichever is greater. Process failure, timeout, missing required telemetry, invalid synchronization, quality-gate failure, or a prespecified nuisance invalidation is recorded with a fixed reason code. The whole pair is excluded if either arm is invalid. Censoring is not imputed. No statistical outlier rule is permitted; extreme valid values remain included. Exclusions may not depend on arm outcomes or effect direction.

All attempted, included, excluded, failed, timed-out, missing, and censored counts must reconcile. If fewer than the fixed final pair count remain, replacement pairs follow the preregistered order generator until the fixed count is reached or 200 attempts are reached. Reaching the bound without enough valid pairs is inconclusive.

### Conclusions

Each workload-endpoint row is `pass` only when its adjusted interval is wholly inside its margin and all evidence gates pass. It is `fail` when the complete interval is wholly outside either equivalence bound or demonstrates a material effect. Otherwise it is `inconclusive`. Missing endpoints are inconclusive.

The overall result is `pass` only if every required workload-endpoint row passes. Any failed or inconclusive row makes the overall result respectively `fail` or `inconclusive`, with failure taking precedence. Conflicting outcomes must be reported together and cannot be selectively summarized.

Development-host evidence is labeled development-only. A target claim requires the declared target hardware and software provenance. Unmatched-host, synthetic-only, incomplete, or uncalibrated evidence cannot support a target low-observer-effect claim.

### Reproducibility record

The baseline records source and ADR revisions; workload commands and input hashes; observer configuration and mode; host OS/kernel; CPU, memory, storage, accelerator and driver/runtime; firmware; power/frequency settings; environment controls; measurement-tool versions; external-harness overhead calibration; sampling period; clock sources; raw-data references and hashes; and analysis implementation/version. Raw paired observations, not summaries alone, are retained in the artifact or in losslessly reproducible content-addressed references.

## Requirement traceability

| Requirement | Contract location |
|---|---|
| R01 | Fixed workloads, matched observer modes, endpoint matrix, and all-endpoints conclusion rule |
| R12 | External authoritative boundaries, asynchronous synchronization, deferred export, and teardown accounting |
| R38 | Paired design, counterbalanced order, precision and multiplicity-adjusted equivalence procedure |
| R40 | Accelerator telemetry, sample loss, collector lateness, GUI SIA GPU activity, provenance, and evidence gates |

## Consequences

This proposed ADR and its empty baseline schema do not establish low observer effect. A claim is permitted only after this proposal is accepted, the procedure is run unchanged on the declared target, complete raw evidence is published, and every required gate and equivalence conclusion passes.