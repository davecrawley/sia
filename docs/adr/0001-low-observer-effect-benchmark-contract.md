# ADR 0001: Low-observer-effect benchmark contract

- Status: Proposed
- Specification revision: `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Companion baseline: `benchmarks/observer-effect/baseline.json`

## Decision

Low observer effect is an equivalence claim, not an absence-of-significance claim. It may be reported only after this preregistered procedure is executed on declared target hardware and every required workload-endpoint conclusion passes. This proposed ADR and its unexecuted baseline do not themselves establish low observer effect.

The unit of analysis is one independent, matched pair of trials. Within a pair, the target executes the same fixed work once with the SIA observation mechanism disabled and once enabled. Inputs, seeds, concurrency, target configuration, host state, measurement tools, and measurement boundaries are identical. Observer mode is the only intended difference.

For ratio endpoints, the pair effect is `log(enabled / disabled)`. The reported estimate and confidence limits are exponentiated ratios. For difference endpoints, the pair effect is `enabled - disabled`. Equivalence requires the complete multiplicity-adjusted confidence interval to lie strictly inside both sides of the predeclared margin. Merely overlapping a margin, failing to reject a difference test, or obtaining a favorable point estimate is insufficient.

## Fixed workloads

All four workload IDs below are required. A campaign must freeze the executable or image digest and input-file digest in the baseline before pilot collection; changing either starts a new campaign and requires a new baseline revision.

| Workload ID | Fixed work and input | Concurrency | Measurement boundary |
| --- | --- | --- | --- |
| `cpu_fixed_work` | One deterministic CPU benchmark invocation completing exactly 10,000 benchmark work units with seed `41041` | One target process and one benchmark worker | Immediately before target start through successful target exit and external-harness synchronization |
| `io_fixed_work` | One deterministic mixed-I/O benchmark over a newly created 8 GiB file, 70% reads, 30% writes, 4 KiB blocks, direct I/O, seed `41042`, completing exactly 1,000,000 operations | One target process, one job, queue depth 1 | Immediately before first submitted operation through completion, flush, and external-harness synchronization |
| `accelerator_fixed_work` | One deterministic accelerator benchmark completing 20,000 fixed-size compute iterations with seed `41043`; no adaptive workload sizing | One target process, one accelerator stream | Before target start through explicit accelerator synchronization and successful target exit |
| `gui_active` | One deterministic scripted GUI replay of 2,000 input events against the same packaged scene and assets with seed `41044` | One GUI target process; event replay is serial | Before application launch through final rendered-frame synchronization, application exit, and observer teardown |

A duration-based substitute, synthetic workload substitution, changed work count, changed seed, or changed concurrency is a protocol change, not another observation in the same campaign. The concrete commands, immutable executable/image identities, and input digests recorded in the baseline are normative for that campaign.

Each arm receives ten minutes of machine idle stabilization followed by one unrecorded workload execution. Observer-enabled warm-up includes SIA initialization. Recorded measurements begin only afterward. No caches are cleared between arms within a pair. Each pair starts from the same declared cache preparation state; pair-to-pair cache state is recorded.

## Required endpoints and margins

Every workload requires `target_elapsed_time_s` and `target_throughput_units_per_s`. The accelerator and GUI workloads additionally require all three accelerator endpoints. Ratios are enabled divided by disabled; the temperature endpoint is an enabled-minus-disabled difference.

| Endpoint | Unit | Pair effect | Symmetric equivalence margin | Aggregation and practical rationale |
| --- | --- | --- | --- | --- |
| `target_elapsed_time_s` | seconds | ratio | 0.99 to 1.01 | Geometric mean of pair ratios; a 1% latency change is the largest operationally negligible slowdown or speedup |
| `target_throughput_units_per_s` | work units/second | ratio | 0.99 to 1.01 | Geometric mean of pair ratios; a 1% capacity change is the largest operationally negligible impact |
| `target_gpu_clock_mhz` | MHz | ratio | 0.99 to 1.01 | Geometric mean of per-trial time-weighted clock means; larger clock displacement can alter accelerator scheduling and performance |
| `target_gpu_power_w` | watts | ratio | 0.98 to 1.02 | Geometric mean of per-trial time-weighted power means; 2% is the largest negligible energy and cooling impact |
| `target_gpu_temperature_c` | degrees Celsius | difference | -1.0 to 1.0 | Arithmetic mean of paired differences of time-weighted means; a 1 °C change is the largest negligible thermal displacement |

Margins are directionally symmetric and may not be widened after pilot or final data are inspected. An overall pass requires equivalence for every one of the 14 required workload-endpoint combinations. A failed, missing, invalid, or inconclusive result prevents an overall pass. Conflicting endpoint or workload outcomes must be reported together and may not be selectively summarized.

Sample loss and collector lateness are mandatory quality gates rather than equivalence endpoints. Every recorded trial must report attempted, received, lost, and late sample counts; sample-loss rate; collector-lateness maximum and p95; and the configured sampling period. A trial fails its quality gate if any field is absent, if sample loss exceeds 0.1%, or if p95 lateness exceeds one configured sampling period. Every `gui_active` trial must additionally record SIA GPU engine time, GPU utilization, GPU memory, and the measurement source. Missing GUI GPU activity evidence makes all GUI conclusions inconclusive.

Every trial also records the complete target elapsed-time or throughput observation, SIA user and system CPU time, peak and time-series RSS, read and write bytes, and, where accelerator-relevant, target GPU clocks, power, and temperature. Raw time series may be stored by a losslessly reproducible content-addressed reference; summaries alone are not a substitute for required raw evidence.

## Pairing, order, and environmental control

Pairs are independent executions. For each workload, order is assigned before collection in balanced blocks of four using seed `731993`: two enabled-first and two disabled-first assignments per block, randomly permuted. An incomplete final block must remain balanced. Order assignments and attempted pair IDs are immutable after generation.

The same isolated machine, boot configuration, CPU set, memory policy, accelerator, power profile, frequency policy, display configuration, driver/runtime, environment variables, input identity, and target build are used in both arms. Network access and unrelated scheduled work are disabled. The external harness records timestamps, cache-preparation state, ambient or inlet temperature when available, CPU and GPU temperatures, frequency/throttling flags, memory pressure, and unexpected background processes.

Order and period effects are estimated from the paired effects using predeclared order and period indicators. A two-sided adjusted 95% interval that excludes zero for either indicator invalidates the affected workload. Any thermal or power throttling, target or collector crash, host suspend, reboot between arms, unexpected background process consuming at least 1% CPU or 1% accelerator for ten seconds, memory pressure event, differing cache-preparation state, or ambient-temperature movement greater than 2 °C within a pair invalidates that pair under the exclusion rules below.

Carryover is checked by comparing the first post-stabilization temperature, cache-state record, and idle resource measurements against their predeclared tolerances. Failure causes pair invalidation before arm outcomes are inspected. Cache-sensitive outcomes are also reported by recorded cache-state stratum; a missing stratum or a stratum lacking the required precision makes the workload inconclusive. No statistical outlier rule is used.

## Uncertainty, multiplicity, and repetition

There are 14 simultaneous required equivalence conclusions. Family-wise confidence is at least 95%. For each endpoint, a two-sided percentile paired-bootstrap interval is formed from 100,000 resamples of whole independent pairs using seed `982451653`. Its per-conclusion confidence level is `1 - 0.05/14` (Bonferroni), providing at least 95% family-wise coverage. Resampling never separates arms or time-series samples within a pair. Pairs are assumed independent; if repeated pairs cannot be made independent because of unresolved period or carryover effects, the affected result is inconclusive.

A separate 12-pair pilot per workload is collected after workload, environment, margins, and analysis code are frozen. Pilot observations never enter final analysis. For each transformed endpoint, let `s` be the pilot standard deviation of paired log effects for ratios or paired differences otherwise, `m` be the corresponding half-margin (`log(1.01)` for the 1% ratios, `log(1.02)` for power, and `1.0` for temperature), `K=14`, and `z(p)` the standard-normal quantile. The required final pair count is:

`n_endpoint = ceil(((z(1 - 0.05/(2K)) + z(0.90)) * s / (m/2))^2)`

The workload count is the maximum `n_endpoint` across its required endpoints, bounded below by 30 and above by 200. If the calculated count exceeds 200, the campaign is infeasible and inconclusive until redesigned under a new preregistration. The numerical precision target is an adjusted 95%-family-wise interval half-width no greater than half the applicable margin.

Exactly the calculated final count is attempted. There is no data-dependent early stopping, extension, or re-estimation. If exclusions leave fewer included pairs or final precision misses the target, the result is inconclusive; a new campaign requires a new preregistration. This fixed rule prevents optional-stopping inflation.

## Attempt accounting and exclusions

Every attempted arm and pair receives a baseline record, including failures and timeouts. Exclusion decisions use only predeclared infrastructure evidence and are made without inspecting the paired arm outcomes. Permitted pair exclusions are: harness or host failure; target or collector crash; timeout at the fixed limit of twice the pilot median target duration; host suspend or reboot; synchronization failure; predeclared interference, throttling, thermal, cache, or carryover violation; corrupt raw artifact; or missing mandatory measurement-quality evidence.

Because equivalence cannot be inferred from selective survivors, any target timeout, censoring, or arm-specific target failure also makes that workload-endpoint conclusion inconclusive even when the pair is excluded. Missing values are never imputed. An anomalous value alone is not excludable. The baseline must account for each attempted run with inclusion status, an enumerated reason, timestamps, and artifact references; included-arm counts must reconcile with attempted, failed, timed-out, censored, missing, and excluded counts.

## Authoritative harness and measurement boundary

An external benchmark harness is authoritative for elapsed time, work count, throughput, trial state, and observer-effect conclusions. SIA may not measure its own performance claim. Harness overhead is calibrated before the pilot with at least 30 empty start/stop trials and recorded, but is not subtracted unless the same preregistered correction is applied identically to both arms.

The enabled boundary includes SIA startup, initialization, steady-state collection, buffering, synchronization, flush/export work, and teardown. Deferred work is not omitted. The disabled arm follows the same harness path and waits at the same synchronization points without running the observation mechanism. Accelerator targets perform an explicit device synchronization before endpoint stop; GUI targets wait for the final presented frame. Host and accelerator timestamps are correlated using the recorded monotonic-clock synchronization procedure. Sampling and timing tools run identically in both arms, and their CPU, memory, I/O, sampling rate, and calibration results are recorded so their observer cost is visible.

## Provenance and permissible claims

Before pilot collection, the baseline must contain exact source and ADR revisions, configuration digest, observer mode definitions, workload commands and input digests, host OS and kernel, CPU and memory, accelerator model, driver and runtime versions, power and frequency settings, isolation and environment controls, harness and measurement-tool versions, clock source, and raw-artifact storage method.

Development-host evidence is labeled `development` and cannot support a target claim. A target claim requires `target` evidence from the declared deployment hardware and software configuration. An unmatched host, synthetic-only substitution for required workloads, incomplete endpoint coverage, missing quality evidence, or uncalibrated estimate is inconclusive and cannot support low-observer-effect wording.

The companion baseline contains deterministic result fields for every required workload-endpoint combination: estimate, adjusted interval, margin, sample count, exclusions, and status. The overall result is mechanically `pass` only when all required result rows are `pass` and all quality and provenance gates pass; it is `fail` if any complete result establishes non-equivalence, and otherwise `inconclusive`. The initial proposed baseline is deliberately `inconclusive` because no measurements have been collected.

## Requirement traceability

| Requirement | Contract coverage |
| --- | --- |
| R01 | Paired external measurement, fixed target work, complete endpoint gating, and claim restrictions |
| R12 | SIA CPU, RSS, I/O, sampling quality, collector lateness, and GUI GPU-activity evidence |
| R38 | Accelerator clocks, power, temperature, synchronization, throttling, and target-hardware provenance |
| R40 | Proposed preregistration, simultaneous equivalence analysis, raw evidence, deterministic results, and reproducibility provenance |