# ADR 0001: Observer-effect benchmark contract

- Status: Proposed
- Specification revision: `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/observer-effect-baseline.json`

## Decision

Low observer effect is established only by paired, target-hardware measurements made under this preregistered contract. The external benchmark harness is authoritative. This proposed ADR and its unexecuted baseline schema are not evidence of low observer effect.

For each workload and endpoint, the unit of analysis is one independent matched pair: one observer-disabled trial and one observer-enabled trial. The arms use the same target build, input, seed, concurrency, duration, host configuration, and environmental controls. They differ only in whether the production observation mechanism is disabled or enabled.

An overall result is `pass` only when every required workload-endpoint combination establishes equivalence and every measurement-quality gate passes. A failed, missing, excluded-without-replacement, or inconclusive combination makes the overall result `inconclusive`, except that a complete adjusted interval outside or crossing a margin produces `fail`. Results may not be selectively summarized to conceal conflicts.

## Requirement trace

| Requirement | Contract provision |
| --- | --- |
| R01 | Paired target measurements, fixed arms, authoritative external harness, and complete provenance |
| R12 | Observer CPU, memory, I/O, GPU activity, sample-loss, and collector-lateness evidence |
| R38 | Target elapsed-time or throughput distributions and accelerator clock, power, and temperature evidence |
| R40 | Preregistered equivalence analysis, family-wise uncertainty control, deterministic conclusions, and prohibition on premature claims |

## Workloads

All workloads use the source revision recorded in the baseline, production compiler settings, one foreground target instance, no unrelated user sessions, networking disabled unless required by the scenario, and a 180-second measurement period following warm-up.

1. `cpu_hash`: one worker repeatedly generates 1 GiB blocks from counter-mode SHA-256 using seed `sia-observer-v1-cpu`, then hashes each block. Exactly 20 blocks are processed. Primary endpoint: external elapsed time.
2. `io_stream`: before either arm, the harness creates an 8 GiB file whose byte at offset `i` is the low byte of SHA-256(`sia-observer-v1-io` concatenated with the unsigned 64-bit block index). Each trial performs direct sequential 4 KiB reads for 180 seconds with queue depth 4 and one worker. Primary endpoint: externally counted bytes per second.
3. `accelerator_matmul`: the target repeatedly multiplies two 8192 by 8192 FP32 matrices for 180 seconds, using seed `sia-observer-v1-gpu`, one process, one accelerator stream, fixed clocks only when the target platform supports fixed clocks, and explicit device synchronization around counting. Primary endpoint: externally counted completed multiplications per second.
4. `gui_active`: the production SIA GUI displays the live overview for 180 seconds at the target display's native mode. At seconds 30, 60, 90, 120, and 150, external automation selects the next overview panel in the fixed order CPU, memory, I/O, accelerator, overview. No synthetic renderer may replace the production GUI. Primary endpoint: the distribution of externally captured presentation latency in milliseconds.

The CPU and I/O generators are controls; `accelerator_matmul` and `gui_active` exercise target accelerator and production GUI paths. A target claim requires all four workloads. Synthetic controls alone cannot support it.

## Endpoints and equivalence margins

Effects are defined so positive values mean greater enabled-arm cost. Ratio effects are analyzed as log ratios and reported as ratios with reciprocal, directionally symmetric bounds. Difference effects use symmetric `[-margin, +margin]` bounds.

| Endpoint | Workloads | Per-pair effect and aggregation | Margin | Practical rationale |
| --- | --- | --- | --- | --- |
| `target_elapsed_ms` | `cpu_hash` | enabled / disabled external elapsed time | `[1/1.03, 1.03]` | A change beyond 3% is user-visible in sustained batch work and exceeds ordinary scheduling tolerance on an isolated host. |
| `target_throughput_per_s` | `io_stream`, `accelerator_matmul` | disabled / enabled externally counted throughput | `[1/1.03, 1.03]` | More than 3% lost productive throughput is operationally material. |
| `target_present_latency_ms` | `gui_active` | enabled / disabled geometric mean of the complete per-frame latency distribution | `[1/1.03, 1.03]` | More than 3% persistent presentation-latency change is material for interactive use. |
| `sia_cpu_percent` | all | enabled minus disabled SIA CPU time divided by arm-boundary wall time, in percentage points | `[-1, +1]` | One continuously consumed percentage point is the largest negligible background CPU cost. |
| `sia_rss_mib` | all | enabled minus disabled peak proportional RSS, MiB | `[-32, +32]` | 32 MiB is a bounded background-memory allowance that remains material on constrained targets. |
| `sia_io_mib_per_s` | all | enabled minus disabled SIA read-plus-write bytes divided by arm-boundary wall time | `[-1, +1]` | Sustained observer I/O above 1 MiB/s may contend with applications and storage. |
| `target_gpu_clock_mhz` | accelerator and GUI | enabled / disabled time-weighted mean target GPU clock | `[1/1.03, 1.03]` | A 3% clock shift can disclose throttling or scheduling interference hidden by target averages. |
| `target_gpu_power_w` | accelerator and GUI | enabled / disabled time-weighted mean board or package power | `[1/1.03, 1.03]` | A persistent power change above 3% is operationally material for thermal and energy budgets. |
| `target_gpu_temperature_c` | accelerator and GUI | enabled minus disabled time-weighted mean temperature, Celsius | `[-2, +2]` | A 2 C shift is large enough to alter thermal headroom while exceeding typical sensor quantization. |
| `sia_gpu_busy_percent` | GUI | enabled minus disabled time-weighted SIA GPU-engine activity, percentage points | `[-1, +1]` | One GPU percentage point is the largest negligible GUI observer load. |

Raw distributions, not quantiles alone, must be retained or referenced by a content-addressed lossless artifact. The baseline enumerates the required combinations and their units.

## Uncertainty and equivalence decision

Let `m` be the number of required workload-endpoint combinations in the baseline. Each combination receives a two-sided BCa paired-bootstrap confidence interval with confidence level `1 - 0.05/m`. At least 100,000 bootstrap resamples are drawn with the pair as the resampling unit and seed `20260904`. Ratios are resampled and interval-constructed on the log scale before exponentiation. Difference endpoints remain on their declared scale. This Bonferroni construction provides at least 95% family-wise coverage across all required conclusions.

Pairs are assumed independent across pair identifiers; observations within a pair are deliberately dependent. The BCa acceleration and bias correction are computed from pair-level effects. If BCa cannot be computed, if fewer than the required pairs remain, or if an interval endpoint is non-finite, the combination is `inconclusive`; no substitute method may be chosen after outcomes are seen.

Equivalence is established only when the complete adjusted confidence interval lies strictly inside the predeclared margin. Touching or overlapping a margin, failing to reject a difference test, or observing a favorable point estimate is insufficient.

## Repetitions and precision

A separate 12-pair pilot is run per workload. Pilot pairs never enter the final analysis. For every endpoint, compute the pilot standard deviation `s` of pair effects on its analysis scale. The target adjusted-confidence half-width is one half of that endpoint's distance from the null to either equivalence boundary. Using `z = Phi^-1(1 - 0.05/(2m))`, calculate `n = ceil((z*s/h)^2)`. The workload's final fixed sample count is the maximum endpoint value, bounded below by 30 independent pairs and above by 100.

The final sample count is fixed and recorded before final collection. There is no interim testing or early stopping. If the calculated count exceeds 100, or 100 valid pairs do not achieve the numerical target, the affected workload is `inconclusive`. This fixed design prevents optional-stopping inflation.

## Pairing, order, and environmental control

Each pair shares one freshly restored machine image and matched input. Pair order is generated before collection from SHA-256(`20260904` plus workload ID plus pair ID); the low bit chooses enabled-first or disabled-first. Orders are balanced within every consecutive block of four pairs, with deterministic inversion of the final choice when needed.

Before each arm, the target receives a 60-second warm-up followed by a 120-second idle stabilization period. A pair starts only when ten-second averages of CPU frequency, GPU temperature, and system load satisfy the target-specific bounds recorded before the pilot. Fan policy, power source, governor, frequency limits, display mode, ambient-temperature band, and process affinity remain fixed.

Order and period are included as preregistered sensitivity covariates. Cache state is restored identically before both arms or explicitly stratified by order. A carryover check compares first-period and second-period effects and the enabled-first and disabled-first strata. A period or order coefficient whose adjusted interval excludes zero, thermal drift beyond 2 C between arm starts, throttling, background CPU above 2%, unexpected I/O, frequency-policy changes, or failure to restore cache state invalidates the pair under the predeclared exclusion rules. If more than 10% of attempted pairs for a workload are invalidated, that workload is inconclusive.

## Measurement boundaries and harness authority

The external harness owns monotonic timestamps, target completion counts, display capture, process accounting, and accelerator telemetry. Its version and calibration are recorded. A no-target calibration run measures harness CPU, I/O, sampling, and timing cost; this cost is reported and must be equivalent across arms, but it is never subtracted selectively.

The arm boundary begins before observer startup and ends only after target completion, device synchronization, observer buffer flush, export, and observer teardown. Deferred work therefore remains in the enabled arm. Target-performance boundaries are separately timestamped inside that arm boundary. Accelerator work is explicitly synchronized before start and after completion. Buffered samples are assigned by source timestamp, and records arriving after teardown count as late rather than disappearing.

## Measurement-quality gates

Every trial records expected, received, late, and lost samples for every collector; sample-loss rate; lateness p50, p95, p99, and maximum; configured period; and collector deadline. GUI-active trials additionally record SIA GPU activity even when its value is zero.

A trial is quality-valid only when sample loss is zero, p99 collector lateness is at most 10% of the configured period, maximum lateness is at most one period, timestamps are monotonic, and all required quality fields exist. Missing fields make every affected workload-endpoint result inconclusive and prevent an overall pass.

## Failures, exclusions, and accounting

Every attempted arm receives a ledger record. Crash, timeout, nonzero target exit, missing arm, missing raw distribution, incomplete teardown, or missing required telemetry invalidates the entire pair and is reported; it is never silently discarded. Timeout is fixed at twice the workload's pilot median duration or 420 seconds, whichever is greater.

Permitted exclusions are limited to the environmental and measurement-quality conditions declared above, machine restart, external power loss, or harness failure identified without comparing arm outcomes. Exclusion decisions use arm-blinded logs and are frozen before effect calculation. Application failures are outcomes, not anomalies, and cause an inconclusive combination unless the contract defines a censored value; this contract defines no censoring or outcome-based outlier removal. Valid replacements use the next preregistered pair ID until the fixed count or maximum attempt bound of 120 is reached.

## Provenance and scope of claims

The baseline records source and ADR revisions, observer configuration, mode, OS, kernel, CPU, memory, accelerator, driver and runtime versions, firmware, power and frequency settings, environment controls, workload identity, inputs and seeds, compiler, harness, measurement tools, and artifact hashes.

Development-host evidence must be labeled `development`. A target low-observer-effect claim requires `target` evidence on the declared target hardware. Unmatched hosts, synthetic-only workloads, incomplete endpoints, uncalibrated tools, missing quality evidence, or estimates substituted for measurement cannot support a target claim.

This ADR remains proposed until human review accepts the workloads, margins, hardware declaration, and analysis. Acceptance of the design still does not constitute a performance claim; only complete target evidence whose recomputed results pass every gate permits the phrase `low observer effect`.