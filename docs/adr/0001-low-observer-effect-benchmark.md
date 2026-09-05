# ADR 0001: Low-observer-effect benchmark contract

- Status: Proposed
- Specification revision: `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/observer-effect-baseline.yaml`

## Decision

Low observer effect is an equivalence claim established from paired benchmark runs. It is not established by this ADR, an empty baseline, failure to detect a difference, a favorable point estimate, or partial endpoint coverage. A claim is permitted only after this proposed contract is accepted, executed unchanged on declared target hardware, and every gate below passes.

The unit of analysis is one independent pair of adjacent runs using the same workload input, seed, concurrency, host configuration, and environmental stratum. One member runs with SIA observation disabled and the other with the production observation mechanism enabled. The enabled arm includes SIA startup, collection, buffering, export, synchronization, and teardown; deferred work may not be moved outside its measurement boundary.

For endpoint `e` in pair `i`, observer effect is the enabled-minus-disabled difference for additive endpoints or a direction-normalized ratio for proportional endpoints. Harmful effects are positive:

- elapsed time and reduced GPU clock: `enabled / disabled - 1`;
- throughput: `1 - enabled / disabled`;
- SIA CPU time, RSS, I/O, GPU activity, GPU power, and GPU temperature: `enabled - disabled`.

Margins are symmetric: equivalence is against `[-margin, +margin]`, even when only one direction is expected to be harmful. Units, aggregation, applicability, and margins are fixed in the baseline artifact. Ratio effects are analyzed on the log scale and transformed back to fractional effects for reporting. Additive effects remain on their declared scale.

## Workloads and controls

The four required workloads are fixed in the baseline artifact:

1. `cpu_compute`: one process and one worker repeatedly hashes deterministic 1 MiB blocks generated from seed 104729 for 600 seconds after a 60-second warm-up.
2. `io_pipeline`: one process sequentially writes and reads an 8 GiB deterministic data set generated from seed 130363 in 64 KiB blocks, calls `fsync` after each GiB, and repeats for 600 seconds after a 60-second warm-up on the declared target filesystem.
3. `gui_interaction`: a 600-second deterministic 1920x1080, 60 Hz GUI trace generated from seed 155921, comprising window creation, text entry, scrolling, image compositing, resize, and close operations in the fixed proportions recorded by the harness manifest, after a 60-second warm-up.
4. `gpu_compute`: a 600-second deterministic accelerator workload generated from seed 196613 using one host submission thread and one device queue, alternating matrix multiplication and image-compositing kernels with synchronization at each one-second epoch, after a 60-second warm-up.

Concurrency is one target workload at a time. No unrelated interactive session, scheduled job, update, network transfer, or power-management transition is allowed. The harness revision, executable and fixture digests, workload proportions, compiler/runtime, display stack, filesystem, and accelerator API must be frozen in provenance before pilot collection. A change requires a new ADR/baseline revision and new data.

These protocols provide repeatable stress coverage, but synthetic evidence alone cannot support a claim about a real deployment. A target claim additionally requires an immutable, representative application trace registered before its pilot, complete endpoint coverage for that trace, and evidence from hardware matching the claimed target. Development-host results, unmatched hosts, uncalibrated estimates, and synthetic-only results must be labeled diagnostic and cannot produce an overall pass.

Enabled and disabled arms differ only in `observer.mode`. Disabled mode must exercise the same target and harness but must not initialize, sample, buffer, export, or tear down SIA collection. Configuration unrelated to that mode is byte-identical.

## Measurement endpoints

The external harness is authoritative for target elapsed time and throughput. It records their per-trial distributions or losslessly reproducible references, rather than only aggregates. SIA self-reported timing cannot establish low observer effect.

Every trial records target elapsed time, target throughput, SIA CPU time, peak RSS, bytes read, bytes written, sample-loss attempted and lost counts, sample-loss rate, collector-lateness samples and p99/max summaries, timestamps, and inclusion status. GUI-active and accelerator trials also record target GPU effective clock, board power, temperature, and SIA GPU activity. GPU samples are aggregated by time-weighted mean after device synchronization; power and temperature additionally retain maxima. CPU time is process user plus system time. RSS is the maximum resident set. I/O is the process-attributed byte count over the complete enabled-arm boundary.

Missing measurement-quality evidence makes every affected workload-endpoint conclusion inconclusive. Sample loss must be at most 0.1%. Collector lateness p99 must be at most twice the configured sampling interval and maximum lateness at most five times that interval. No imputation is allowed.

The external harness cost is measured in observer-disabled calibration runs with and without harness sampling. Calibration uses the same sample interval and endpoint set. Median timing cost must be below 0.5% and its family-wise interval upper bound below 1%; otherwise the campaign is invalid until the harness is recalibrated. Harness cost is applied equally and is never subtracted post hoc. Monotonic clocks are used. CPU/GPU clock domains are synchronized at each trial boundary. Accelerator work is explicitly synchronized before target completion. Buffered export and observer teardown finish before enabled elapsed time stops.

## Equivalence and uncertainty

For each required workload-endpoint combination, report the mean paired effect and a simultaneous two-sided confidence interval. Confidence is at least 95% family-wise across all required conclusions. Let `K` be the number of required combinations fixed before the pilot. Each interval uses Bonferroni confidence `1 - 0.05/K`, with equal tail allocation.

Intervals are percentile intervals from 100,000 deterministic stratified paired-bootstrap resamples using seed 32452843. The resampling unit is the entire independent pair; pairs are resampled within workload and order sequence, preserving all endpoints from a pair together. This assumes independence between pairs and exchangeability within each declared stratum. Autocorrelation or session clustering requires resampling whole independent sessions and a preregistered amendment before confirmatory collection.

Equivalence is established only when the complete adjusted interval lies strictly inside `[-margin, +margin]`. Touching or overlapping a margin, an unadjusted interval, failure to reject a difference test, or a favorable point estimate is insufficient. Failed, missing, invalid, or inconclusive combinations prevent an overall pass. Conflicting endpoint outcomes must be reported together and cannot be selectively summarized.

Margins in the baseline are tied to practical impact: 3% bounds changes in target latency, throughput, or GPU clock before they become user-visible or materially reduce capacity; 300 ms of SIA CPU per 60 seconds limits collection to half of one core; 64 MiB RSS avoids meaningful memory-pressure changes on the minimum target; 5 MiB of read or written I/O per trial avoids storage churn; 5 W and 2 C limit accelerator power and thermal interference; and one percentage point limits SIA GPU occupancy. Margin changes require a new proposed ADR before any affected pilot.

## Pairing, order, and environmental validity

Pairs use matched inputs and adjacent periods. Order is counterbalanced in deterministic blocks of four pairs as AB, BA, BA, AB, where A is disabled and B is enabled; block order is generated from seed 49979687. The first arm in each pair is selected from that schedule before execution.

Warm-up is excluded from endpoint aggregation but fully recorded. Before each arm, caches are returned to the declared campaign state; either both arms use a documented cold-cache reset or both use a separately named warm-cache stratum. Mixing states is forbidden. There is a five-minute quiescent interval between arms and a fifteen-minute interval between pairs.

Each campaign records arm order, period, cache state, ambient temperature, CPU frequency, GPU frequency, power, target temperature, throttling flags, load average, and unrelated process CPU/I/O. A preregistered linear sensitivity model `effect ~ arm + period + sequence + cache_stratum` is reported beside the paired analysis. Any arm is objectively invalid if warm-up is incomplete; cache state differs within its pair; thermal or power throttling occurs; ambient temperature differs by more than 2 C within a pair; background CPU exceeds 5% of one core for over 10 seconds; unrelated I/O exceeds 100 MiB; clock synchronization fails; or a required collector/harness health check fails. More than 10% invalid pairs, a significant order or carryover term at the Bonferroni-adjusted 5% level, or a material period trend exceeding half an equivalence margin invalidates the workload campaign rather than being adjusted away.

## Repetition and stopping

The pilot consists of 10 independent pairs per workload and is excluded from confirmatory analysis. Using the pilot standard deviation `s` of each paired effect and its margin `delta`, the required confirmatory size is

`ceil(((z(1 - 0.05/(2K)) + z(0.90)) * s / delta)^2)`.

The workload sample size is the maximum over its required endpoints, bounded below by 30 independent pairs and above by 200. If the calculated size exceeds 200, the campaign is inconclusive. The numerical precision target is a family-wise adjusted interval half-width no greater than the applicable margin, at 95% family-wise confidence and 90% planning power at zero true effect. The frozen sample size is recorded before confirmatory data collection. There is no interim equivalence analysis, optional stopping, early success, or sample-size re-estimation. A replacement for an objectively invalid run is a new attempted pair and does not erase the invalid attempt.

## Failures, exclusions, and accounting

Every attempted arm and pair receives an identifier and remains in the baseline. Timeout is fixed at 120% of nominal workload duration plus teardown. Crashes, timeouts, missing values, device resets, and observer failures are not excluded as outliers: they make affected conclusions inconclusive and are counted. Right-censored values retain their bound and censoring reason; they are not imputed.

Objective environmental invalidations listed above may be excluded only from both arms of the pair and only from evidence recorded without inspecting arm outcomes. Anomaly flags use a frozen median-absolute-deviation rule solely for sensitivity reporting and never remove observations from the primary analysis. The artifact reports attempted, included, excluded, failed, timed-out, missing, and censored counts. Post-outcome exclusion and selective reruns are forbidden.

## Provenance and conclusions

The baseline records source and ADR revisions; configuration digest and observer mode; host OS and kernel; CPU and memory; accelerator, driver, and runtime; power and frequency settings; environment controls; workload/input identity; harness and measurement-tool versions; timestamps; raw paired observations; and reproducible artifact references.

Each workload-endpoint result records its estimate, adjusted interval, margin, sample count, exclusions, and deterministic `pass`, `fail`, or `inconclusive` status. The overall result is `pass` only if all required combinations pass and all quality, provenance, representativeness, target-hardware, and calibration gates pass. A complete interval outside or crossing a margin is `fail`; missing evidence or an interval that cannot establish equivalence is `inconclusive`. Both prevent an overall low-observer-effect claim.