# ADR 0001: Observer-effect measurement contract

- Status: Proposed
- Specification revision: 4479bcc19db72f6ad243a87e4b7271496d60d0b7
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/observer-effect/baseline.yaml`

## Decision

Observer effect is measured by paired runs of the same workload on the same target, once with SIA observation disabled and once with it enabled. The binary, configuration, environment, workload input, and measurement boundary are identical between arms; only the observation mechanism under evaluation changes.

A low-observer-effect conclusion is an equivalence conclusion, not a failure to find a difference. Every required workload-endpoint combination must establish equivalence. A failed, missing, invalid, or inconclusive combination makes the overall conclusion inconclusive or failed; favorable combinations may not be selected or summarized as an overall pass.

This proposed contract and its unexecuted baseline do not constitute evidence of low observer effect. Such a claim is permitted only after the preregistered procedure is run on declared target hardware and all gates pass.

## Unit of analysis and effect

The independent unit is one matched pair containing one disabled and one enabled trial. Trial-level samples are reduced using the endpoint aggregation rules below before the paired effect is calculated. Samples within a trial are not independent repetitions.

For positive ratio endpoints, the effect is `log(enabled / disabled)`. The symmetric equivalence interval is `[-log(1 + m), +log(1 + m)]`; reported ratios are obtained by exponentiation. For difference endpoints, the effect is `enabled - disabled` and the interval is `[-m, +m]` in the stated unit. These definitions treat improvements and regressions symmetrically.

Equivalence is established only when the complete multiplicity-adjusted confidence interval lies strictly inside both equivalence limits. Margin overlap, a favorable point estimate, or failure to reject a difference test is insufficient.

## Required workloads

All workload implementations are versioned as harness protocol `sia-observer-v1`. Each timed trial lasts 600 seconds after warm-up.

1. `cpu_hash_v1`: `logical CPU count minus one`, with a minimum of one, independent workers repeatedly hash deterministic 64 MiB buffers generated from hexadecimal seed `5349412d4350552d5631`. Concurrency and generated buffers remain fixed across a pair. The target endpoint is completed bytes per second.
2. `storage_stream_v1`: one worker performs sequential 4 MiB direct reads from a pre-created 64 GiB file generated from hexadecimal seed `5349412d494f2d5631`. The file checksum is recorded, the page cache is dropped by the isolated harness before both arms, and the target endpoint is completed bytes per second.
3. `gpu_compute_v1`: one process repeatedly executes the harness's versioned deterministic Mandelbrot FP32 kernel at 8192 by 8192 pixels with seed `5349412d4750552d5631`. It uses one accelerator queue, synchronizes after every iteration, and reports completed pixels per second.
4. `gui_replay_v1`: the SIA GUI is active while the harness replays the versioned ten-minute `gui-replay-v1` input trace at 60 Hz and seed `5349412d4755492d5631`. Rendering is synchronized at every frame. The target endpoint is presented frames per second.

The exact harness revision, input checksums, resolved CPU count, accelerator identity, and effective configuration must be recorded before collection. Synthetic workloads characterize only these fixed workloads and cannot support broader target claims without separately preregistered representative workloads.

## Required endpoints and margins

The following trial aggregations and paired margins are fixed. A time-weighted statistic uses sample residence time and excludes no valid sample.

| Endpoint | Required workload | Unit and trial aggregation | Effect and symmetric margin | Practical rationale |
| --- | --- | --- | --- | --- |
| `target_throughput` | all | operation-specific units per second over the authoritative external boundary | log ratio, ±log(1.02) | A two-percent change is the largest operationally negligible target impact. |
| `sia_cpu_core_fraction` | all | SIA process CPU seconds divided by boundary seconds | difference, ±0.005 cores | Half of one percent of a continuously occupied core is the CPU budget. |
| `sia_peak_rss_mib` | all | maximum SIA resident set, MiB | difference, ±16 MiB | Sixteen MiB is the smallest material resident-memory budget increment. |
| `sia_io_mib_s` | all | SIA read plus write bytes divided by boundary seconds, MiB/s | difference, ±1 MiB/s | One MiB/s is the negligible sustained storage-traffic budget. |
| `target_gpu_clock_mhz` | GPU and GUI | time-weighted mean target GPU core clock, MHz | difference, ±30 MHz | Thirty MHz is below one normal performance-state step on the target class. |
| `target_gpu_power_w` | GPU and GUI | time-weighted mean board power, W | difference, ±3 W | Three watts is the target measurement and operational relevance threshold. |
| `target_gpu_temperature_c` | GPU and GUI | maximum steady-state temperature, degrees C | difference, ±2 degrees C | Two degrees is the threshold for a meaningful cooling or throttling effect. |
| `sia_gpu_busy_pct` | GUI only | time-weighted mean SIA GPU-engine activity, percentage points | difference, ±1 percentage point | One point is the negligible GUI observer GPU budget. |

This produces 23 required conclusions. Accelerator endpoints are inapplicable only to the CPU and storage workloads. Inapplicability is fixed here and cannot be chosen after measurement.

## Uncertainty and multiplicity

The primary estimate is the equal-weight mean of paired effects, stratified by the two arm sequences. Within each sequence, pairs are sampled with replacement using a studentized paired bootstrap; the two sequence means are then equally weighted. Use 20,000 bootstrap replicates and deterministic seed `sha256(session_id + specification_revision + endpoint_id)`.

For each of the 23 conclusions, use two-sided Bonferroni tail probability `0.05 / (2 * 23)`, yielding an individual confidence level of 99.7826 percent and at least 95 percent family-wise coverage by the union bound. The resampling unit is the complete independent pair. Assumptions are independence between pairs, exchangeability within randomized sequence, stable endpoint definitions, and valid pairing. An interval that cannot be constructed, including because of insufficient variation or observations, is inconclusive.

## Repetition and precision

Each workload first receives 20 pilot pairs. Pilot observations are recorded but never enter the final estimate. Before final collection, the harness calculates the required final pair count for every endpoint as

`ceil((z(1 - 0.05 / (2 * 23)) * pilot_sd / (0.5 * margin))^2)`.

The workload count is 110 percent of the largest endpoint count, rounded upward and clamped to a minimum of 30 and a maximum of 200 independent final pairs. The numerical precision target is an adjusted 95-percent-family-wise interval half-width no greater than half the equivalence margin. If the calculated count exceeds 200, or the completed interval misses the precision target at 200 pairs, the affected result is inconclusive.

There is no early success stop and no repeated testing. The final count is fixed before final outcomes are inspected. Collection stops at the fixed count or 200 attempted final pairs, whichever occurs first. Pilot exclusion, a single final analysis, and the fixed maximum prevent optional-stopping inflation.

## Pairing, order, and environmental control

Inputs, seeds, concurrency, duration, warm-up, machine state, binary, configuration, and measurement tools are matched within each pair. Enabled mode activates the evaluated collectors and GUI state declared by the workload. Disabled mode runs the same SIA binary and lifecycle with those collectors disabled.

Arm order is generated before measurement in balanced blocks of four using `sha256(session_id + workload_id + pair_id)`. Each block contains two enabled-first and two disabled-first pairs. Order assignments and all attempts are retained.

Before each arm there is a ten-minute idle stabilization period followed by a workload-specific two-minute untimed warm-up. CPU and GPU temperature must return within 2 degrees C of the pair's preregistered starting baseline. Storage cache state is reset identically. Power and frequency policy, affinity, priority, display state, and network policy remain fixed.

The harness records arm order, period, start and end timestamps, warm-up state, cache-reset outcome, CPU and GPU temperatures, throttle flags, background CPU, memory pressure, swap, and unexpected I/O. A pair is environmentally invalid only when a predeclared condition occurs: any thermal or power throttle flag; cache-reset failure; background CPU above 2 percent for more than five consecutive seconds; any swap activity; memory pressure; unexpected external I/O above 5 MiB/s for five seconds; or failure to return within the temperature bound. The decision is made from harness diagnostics without inspecting arm endpoint outcomes, and invalidates both arms.

A cooldown sentinel is recorded before each arm. A temperature difference above 2 degrees C, background-load difference above 2 percentage points, or failed cache reset is treated as carryover and invalidates the pair. Sequence-stratified estimates address order; period trends are reported by regressing paired effects on pair index and sequence. A trend across the run exceeding half an endpoint margin, or a sequence contrast exceeding half a margin, makes that endpoint inconclusive rather than being corrected after inspection.

## Measurement boundary and quality

The external benchmark harness is authoritative. Its boundary begins immediately before SIA and target startup and ends only after target synchronization, observer buffering and export, collector flush, and SIA teardown finish. Deferred enabled-arm work is therefore included. Accelerators are explicitly synchronized before timestamps and shutdown. The same harness lifecycle and no-op calls occur in the disabled arm.

Before final collection, the harness timing path is calibrated with 100 empty paired runs. Its median arm cost and sampling CPU cost are recorded and must each be below ten percent of the smallest applicable practical margin. Calibration is not subtracted from outcomes because it is matched; a failed calibration makes results inconclusive.

Every arm records expected and received collector samples, loss count and rate, scheduled and actual timestamps, lateness p50, p95 and maximum, late-sample count, collector errors, and measurement-tool synchronization. Sample loss above 1 percent, any collector error, or maximum lateness above two sampling periods invalidates measurement quality for the affected pair. Missing quality fields make all affected workload-endpoint results inconclusive. Every GUI-active arm additionally records `sia_gpu_busy_pct`; omission makes the GUI workload inconclusive.

## Failures, exclusions, and accounting

Every attempted pilot and final run is assigned a monotonically increasing attempt ID and retained, including failed, timed-out, missing, censored, and excluded runs. Timeout is 750 seconds from boundary start.

Harness or environmental failures meeting the objective rules above exclude the complete pair. A target crash, SIA crash, enabled-only failure, timeout, missing endpoint, or censoring is not treated as a removable outlier: it makes the affected workload inconclusive. No statistical outlier rule is used. Exclusion rules cannot be added or changed after arm outcomes are viewed. Excluded pairs are not silently replaced; collection continues only until the fixed count or 200 attempted pairs. Counts by arm, reason, phase, and workload are reported.

## Provenance and claim scope

Each session records source revision, ADR revision, specification revision, full configuration and observer mode, host OS and kernel, CPU and memory, accelerator model, driver and runtime, firmware, power and frequency settings, isolation and environment controls, workload and input checksums, randomization seed, and all harness and measurement-tool versions.

Evidence is labeled `development-host` or `target-hardware`. Development-host evidence may debug the method but cannot support a target low-observer-effect claim. A target claim is forbidden for an unmatched host, synthetic-only evidence presented beyond its declared scope, incomplete endpoint coverage, uncalibrated harness, missing raw data, or missing provenance.

## Deterministic decision procedure

For every preregistered workload-endpoint row:

1. Verify provenance, calibration, attempt accounting, quality fields, final sample count, and precision.
2. Calculate arm-level aggregates and the paired effect on the declared scale.
3. Construct the adjusted interval using the fixed bootstrap procedure.
4. Mark `pass` only if the complete interval lies strictly inside both margin limits.
5. Mark `fail` when a valid complete interval crosses or lies outside a limit.
6. Mark `inconclusive` when required evidence or validity conditions are absent or invalid.

The overall result is `pass` only if all 23 rows pass. It is `fail` if any row fails and none is inconclusive; otherwise it is `inconclusive`. The baseline stores estimates, adjusted bounds, margins, pair counts, exclusions, and statuses so an independent agent can recompute this decision without discretion.

## Requirement traceability

| Requirement | Contract location |
| --- | --- |
| R01 | Paired observer definition, fixed modes, boundaries, and all-endpoint decision rule |
| R12 | CPU, RSS, I/O, timing, sample-loss, and collector-lateness evidence |
| R38 | Accelerator clocks, power, temperature, synchronization, and SIA GPU activity |
| R40 | Reproducibility provenance, raw paired evidence, uncertainty, and target-claim restrictions |