# ADR 0001: Observer-effect benchmark contract

- Status: Proposed
- Specification revision: `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/observer-effect-baseline.json`

## Context

SIA observes a target process and may itself change target performance or accelerator behavior. A low-observer-effect claim therefore requires target-hardware measurements under a fixed, independently reviewable contract. This ADR preregisters that contract; its existence and an unexecuted baseline do not establish low observer effect.

R01 requires useful target observation, R12 requires measurement-quality evidence, R38 requires observer-effect characterization, and R40 requires reproducible evidence. This decision traces all four requirements to fixed workloads, paired measurements, quality gates, provenance, and deterministic conclusions.

## Decision

### Claim and unit of analysis

The unit of analysis is one independent pair of runs of one workload on one target host. A pair contains the same input, seed, concurrency, duration or work count, environment, and measurement boundaries. Its arms differ only in whether the SIA observation mechanism under evaluation is disabled or enabled.

For a strictly positive endpoint, observer effect is the paired log ratio

`d = log(enabled / disabled)`.

The reported estimate and interval are exponentiated ratios. For endpoints naturally measured relative to zero, observer effect is the paired difference

`d = enabled - disabled`.

All margins are directionally symmetric: ratio margins are reciprocal on the log scale and difference margins are `[-M, +M]`. Improvement in one direction does not excuse harm in the other.

### Required workloads

The following workload IDs are required. Implementations may add workloads but may not remove, rename, or substitute these workloads after data inspection.

| ID | Fixed work and input | Concurrency | Timed boundary | Warm-up | Required endpoint set |
| --- | --- | ---: | --- | --- | --- |
| `cpu_fixed_work` | 600 s deterministic CPU benchmark, input `cpu-v1`, seed 104729 | 1 target worker | Immediately before first benchmark operation through completion of the fixed operation count | One untimed 60 s execution before each arm | common |
| `storage_fixed_work` | Sequential write, sync, cold-cache read, and verify of a seeded 16 GiB file, input `storage-v1`, seed 130363 | 1 target worker, queue depth 1 | Before file creation through verification and final sync | One untimed 2 GiB cycle before each arm; benchmark file recreated | common |
| `gpu_fixed_work` | 600 s fixed-operation accelerator benchmark, input `gpu-v1`, seed 155921 | 1 process and 1 accelerator stream | Before first submission through final device synchronization | One untimed 60 s execution before each arm | common plus accelerator |
| `gui_active_fixed_script` | 600 s deterministic GUI replay `gui-v1`, seed 196613, fixed window size and frame cadence | 1 application and 1 replay driver | Before first scripted event through final rendered-frame synchronization and application quiescence | One untimed complete replay before each arm | common plus accelerator and GUI |

The baseline must replace workload tool, input digest, operation count, resolution, cadence, and target executable placeholders before collection. Any such replacement creates a new ADR revision and invalidates data collected under the earlier definition.

### Endpoints, aggregation, and equivalence margins

Every endpoint below is required wherever its set applies. Raw observations retain the stated unit. Each estimate is the arithmetic mean of paired differences, except ratios, whose estimate is the geometric mean of paired ratios. Quantiles describe the arm-labeled raw trial distribution and are not substitutes for paired inference.

| Endpoint ID | Set | Unit | Effect | Equivalence margin | Practical rationale |
| --- | --- | --- | --- | --- | --- |
| `target_elapsed_ms` | common | ms | ratio | 0.97 to 1.030927835 | A 3% timing change is the largest operationally negligible shift. |
| `target_throughput_per_s` | common | operations/s | ratio | 0.97 to 1.030927835 | A 3% sustained-rate change is the largest operationally negligible shift. |
| `sia_cpu_core_seconds` | common | core-s | difference after division by trial wall seconds | -0.02 to +0.02 cores | Two percent of one core is the CPU budget attributable to observation. Disabled-arm SIA use is zero and must still be recorded by the harness. |
| `sia_rss_peak_mib` | common | MiB | difference | -64 to +64 MiB | 64 MiB is the accepted resident-memory budget. |
| `sia_read_mib` | common | MiB | difference | -16 to +16 MiB | 16 MiB per trial is the accepted observer-read budget. |
| `sia_write_mib` | common | MiB | difference | -16 to +16 MiB | 16 MiB per trial is the accepted observer-write and export budget. |
| `collector_sample_loss_rate` | common | fraction | difference | -0.005 to +0.005 | Half a percentage point is the maximum negligible loss shift; each arm must also have absolute loss at most 0.005. |
| `collector_lateness_p95_ms` | common | ms | difference | -1 to +1 ms | One millisecond is the maximum negligible p95 scheduling-lateness shift; each arm must also be at most 5 ms. |
| `target_gpu_clock_mhz` | accelerator | MHz | ratio | 0.97 to 1.030927835 | A 3% clock change can affect delivered accelerator work. |
| `target_gpu_power_w` | accelerator | W | ratio | 0.97 to 1.030927835 | A 3% power change is the largest negligible energy/thermal perturbation. |
| `target_gpu_temperature_c` | accelerator | degrees C | difference | -2 to +2 degrees C | A 2 degree shift is the accepted thermal-noise envelope. |
| `sia_gpu_utilization_pct` | GUI | percentage points | difference | -1 to +1 percentage points | More than one percentage point of SIA-attributed GPU activity is operationally material. |

A workload-endpoint result passes only when the complete multiplicity-adjusted confidence interval is strictly inside its margin and all data-quality gates pass. Merely overlapping a margin, failing to reject a difference test, or observing a favorable point estimate is insufficient.

Overall low observer effect passes only when every required workload-endpoint combination passes. A failed, missing, invalid, or inconclusive result makes the overall conclusion non-passing. Results must be reported as a complete matrix; conflicting outcomes may not be selectively summarized.

### Pairing, order, and environment

Pairs are executed in blocks of two. A recorded PRNG seed deterministically assigns `disabled-enabled` or `enabled-disabled` order with equal allocation within each workload; an odd final pair uses the seeded assignment. Inputs, affinity, priority, concurrency, power source, power limit, frequency policy, display configuration, filesystem, free space, and network policy are identical across arms.

The host is dedicated to the experiment. Automatic updates, indexing, backups, scheduled jobs, dynamic display changes, and unrelated user sessions are disabled. Ambient temperature is recorded. A run starts only after CPU and accelerator temperatures remain within 2 degrees C of the workload-specific starting band for five minutes.

Warm-up is performed separately before each arm and is never analyzed. The declared cache state is recreated before each arm. Any thermal-throttling flag, frequency/power-policy change, target restart, harness failure, or unexpected background CPU above 5%, disk throughput above 10 MiB/s, or accelerator utilization above 2% during the 60 s preflight invalidates the entire pair without regard to arm outcomes.

Order and period are included as fixed effects in a sensitivity regression over paired effects. Cache state and starting-temperature band are strata. Carryover is tested by the order-by-arm term. If an adjusted 95% interval for an order, period, cache-stratum, or carryover effect excludes zero and its magnitude exceeds half the endpoint equivalence margin, that workload-endpoint is inconclusive unless the effect was already included in the preregistered primary model. Thermal throttling always invalidates the pair rather than being modeled.

### Repetitions and precision

The confidence level is at least 95% family-wise. Each workload begins with 20 independent pilot pairs. Pilot pairs enter the final analysis if they satisfy the same preregistered collection and inclusion rules.

For each endpoint, let `s` be the pilot standard deviation of paired log ratios or differences, `K` the total number of required workload-endpoint conclusions, and `c = Phi^-1(1 - 0.025/K)`. The desired half-width `h` is half that endpoint's log-scale or difference-scale equivalence-margin half-width. The fixed final sample count is

`n = min(200, max(30, ceil((c*s/h)^2)))`.

The workload receives the maximum `n` required by any of its endpoints. This calculation is performed once after exactly 20 pilot pairs, recorded in the baseline, and not revised. Collection then continues to that fixed count. There is no repeated significance testing or early success stop. If 200 pairs do not attain the numerical precision target, or fewer than 30 valid independent pairs remain, the affected results are inconclusive. This fixed two-stage rule prevents optional stopping from inflating type-I error.

### Uncertainty and multiplicity

Primary intervals are simultaneous two-sided 95% studentized max-|t| bootstrap intervals. The resampling unit is the independent pair. Within each workload, complete pair vectors are resampled jointly within order/cache strata so endpoint dependence is retained. Workloads are resampled independently. The bootstrap uses 100,000 replicates and recorded seed 32452843. In each replicate, the maximum absolute studentized deviation across every required workload-endpoint conclusion supplies the single critical value. This provides at least 95% family-wise coverage for the complete required matrix.

This procedure assumes independent pairs, exchangeability within declared strata, finite paired-effect variance, and stable workload/environment definitions. Zero or negative values for ratio endpoints violate the method and make that result inconclusive. A degenerate variance, fewer than 30 valid pairs, an uncomputable bootstrap, or a failed diagnostic is inconclusive rather than a pass.

### Measurement boundaries and harness authority

An external benchmark harness, not SIA timestamps, is authoritative. The same harness and sampling schedule run in both arms. Its calibration records null-loop timing and sampling cost before the experiment; a calibration exceeding 0.5% of trial duration or differing between arms invalidates the workload. Calibration is reported but not subtracted.

Enabled-arm timing begins before observer initialization and ends only after final sampling, buffer drain, export, flush, and observer teardown. Deferred work may not be moved outside the enabled boundary. Disabled runs execute a matched no-op lifecycle boundary. The harness uses a monotonic clock, records wall-clock UTC timestamps, and synchronizes clocks where multiple devices are involved. GPU endpoints require explicit device synchronization before start and end reads. Buffering, asynchronous accelerator work, export, and teardown are therefore included rather than hidden.

### Run accounting and data quality

Every attempted run receives an immutable attempt ID before execution. The baseline records success, failure, timeout, censoring, inclusion, and reason. A timeout is the fixed workload duration plus 20%; timed-out and failed arms invalidate their whole pair and remain in the ledger. Missing endpoint or quality data makes the affected workload-endpoint inconclusive.

Exclusion is allowed only for the predeclared environment violations above, harness corruption, target crash, power loss, or input-integrity failure. The rule is evaluated without comparing arm outcomes and excludes both members of a pair. Performance outliers are never excluded merely for being extreme; robust sensitivity summaries may accompany but never replace the primary result. Censored values are retained with censor bounds, and any endpoint lacking a preregistered censoring model is inconclusive.

Every trial records expected and received samples, sample-loss count and rate, collector lateness p50/p95/max, and late-sample count. Every GUI-active trial also records SIA GPU active time, utilization, engine, and process-attribution method. Missing quality fields make all affected trial endpoints inconclusive and prevent an overall pass.

### Baseline and provenance

The machine-readable baseline contains the schema, endpoint matrix, attempted-run ledger, arm-labeled raw observations, results, and overall conclusion. Raw values or content-addressed losslessly reproducible references are mandatory; summaries alone are insufficient.

Before collection, provenance fields must identify source and ADR revisions, full configuration and observer mode, target executable and input digests, host OS/kernel, CPU, memory, accelerator and driver/runtime, firmware, power/frequency settings, isolation controls, measurement tools and versions, workload identity, and time source. Any material provenance mismatch splits a stratum and requires a separately powered preregistration.

Development-host evidence is labeled `development` and cannot support a target claim. Target low-observer-effect claims require the declared target hardware, representative fixed workloads, complete endpoint coverage, and calibrated external measurement. Unmatched hosts, synthetic-only evidence, uncalibrated estimates, or incomplete data cannot establish the claim.

## Consequences

The contract favors falsifiability over a convenient aggregate score. It can produce an inconclusive result even when most point estimates look favorable. Changes to workloads, margins, exclusions, sampling, or analysis after outcomes are visible require a new proposed ADR and new data.

No claim is permitted while this ADR is proposed or the baseline is unexecuted. A claim becomes eligible only after this ADR is accepted, the preregistered procedure runs on the declared target, all required evidence is present, and every result and the overall conclusion pass.