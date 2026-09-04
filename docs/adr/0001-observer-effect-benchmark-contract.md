# ADR 0001: Observer-effect benchmark contract

- Status: Proposed
- Specification revision: `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Machine-readable baseline: `benchmarks/observer-effect/baseline.json`

## Decision

Low observer effect is a statistical equivalence claim. It is established only by the preregistered paired experiment below, performed on the declared target hardware, with complete raw observations and all required conclusions passing. This proposed ADR and its unexecuted baseline do not themselves establish low observer effect.

The observer effect for a pair is the difference between otherwise identical observer-enabled (`E`) and observer-disabled (`D`) runs. Additive endpoints use `E-D`. Strictly positive scale endpoints use `log(E/D)` and are reported as `E/D`. One independent, matched enabled/disabled pair is the unit of analysis. Repeated samples within a trial are reduced to the endpoint aggregation declared below and are not independent observations.

An overall pass is permitted only if every required workload-endpoint result is `pass` and every measurement-quality gate passes. A failed, missing, invalid, or inconclusive result prevents an overall pass. Conflicting results must be reported individually and may not be selectively summarized.

## Traceability

| Requirement | Contract provision |
|---|---|
| R01 | External paired measurements, fixed boundaries, equivalence margins, and an all-endpoints pass rule make low observer effect falsifiable. |
| R12 | Observer CPU time, RSS, I/O, sample loss, collector lateness, and configuration are recorded for every trial. |
| R38 | Target GPU clock, power, and temperature are required endpoints; GUI trials additionally require SIA GPU-activity evidence. |
| R40 | Raw paired evidence, provenance, uncertainty, exclusions, and deterministic conclusions are retained in the baseline artifact. |

## Frozen workloads

The harness exposes the following immutable workload IDs. Its workload manifest, implementation revision, command vector, input hashes, and tool versions must be copied into the baseline before the first pilot. Changing any of them starts a new campaign and baseline revision.

| Workload | Fixed execution |
|---|---|
| `fixed_job` | One deterministic target job using seed `104729`, concurrency 1, a 60 s untimed warm-up, then one complete timed job. Elapsed time is measured from target release until completion; throughput is completed canonical work units divided by that interval. |
| `sustained_throughput` | Deterministic input stream using seed `130363`, concurrency equal to the declared target production concurrency, 60 s untimed warm-up, then 600 s measurement. The exact admitted input sequence is replayed in both arms. |
| `gui_active` | The declared target GUI application, deterministic interaction replay `gui-replay-v1` using seed `155921`, concurrency 1, 60 s untimed warm-up, then 600 s measurement. The window size, display refresh, compositor, and replay event times are fixed and recorded. |

A campaign is not executable until the baseline contains losslessly reproducible workload-manifest references and concrete input identities. Synthetic workloads may support development but cannot replace any workload above for a target claim.

## Required endpoints and margins

Every endpoint below is required for every workload. Ratios use a directionally symmetric margin on the log scale. Additive margins are symmetric about zero. The same margins apply to all three workloads; therefore the family contains 27 required workload-endpoint conclusions.

| Endpoint | Trial value and unit | Effect | Equivalence margin | Practical rationale |
|---|---|---|---|---|
| `target_elapsed_time` | External monotonic elapsed time, ms | `E/D` | `[1/1.03, 1.03]` | A change exceeding about 3% is operationally noticeable for fixed work. |
| `target_throughput` | Completed canonical work units/s | `E/D` | `[1/1.03, 1.03]` | A change exceeding about 3% materially changes capacity. |
| `sia_cpu_time` | SIA process CPU ms per measured wall-second | `E-D` | `[-0.50, 0.50]` | More than 0.5 ms/s is over 0.05 of one CPU core. |
| `sia_rss` | Maximum SIA resident set size, MiB | `E-D` | `[-32, 32]` | A 32 MiB shift is the largest accepted memory-budget change. |
| `sia_io` | SIA read plus write throughput, KiB/s | `E-D` | `[-64, 64]` | Sustained I/O above 64 KiB/s can affect constrained systems. |
| `target_gpu_clock` | Time-weighted mean target GPU core clock, MHz | `E/D` | `[1/1.03, 1.03]` | A 3% clock displacement may indicate altered accelerator scheduling or throttling. |
| `target_gpu_power` | Time-weighted mean target GPU board power, W | `E/D` | `[1/1.05, 1.05]` | A 5% power displacement is practically material. |
| `target_gpu_temperature` | Maximum target GPU temperature, °C | `E-D` | `[-2, 2]` | A 2 °C displacement can affect fan or throttle policy. |
| `target_gpu_activity` | Time-weighted mean target GPU busy time, percent | `E-D` | `[-1, 1]` percentage points | A one-point activity displacement is the accepted scheduling-noise budget. |

For ratio endpoints, equivalence means the entire adjusted interval for `log(E/D)` lies strictly inside `[-log(U), log(U)]`, where `U` is the upper margin. Reported ratio limits are exponentiated. For additive endpoints, the entire adjusted interval must lie strictly inside `[-Δ, Δ]`.

Margin overlap, failure to reject a difference test, or a favorable point estimate is insufficient. Equality with a margin boundary is not a pass.

## Pairing, order, and environmental control

Each pair uses the same host, boot image, target revision, workload manifest, input sequence, seed, concurrency, duration, power policy, frequency policy, environment, and measurement tooling. The arms differ only in whether the SIA observation mechanism under evaluation is disabled or enabled. Disabled mode keeps non-observation setup common to both arms.

Pair order is balanced within each workload. For even pair IDs, order is generated by a deterministic permutation using campaign seed `8675309`; each adjacent pair of pair IDs contains one `DE` and one `ED` order. Workloads are run in deterministic randomized blocks. No pair may contain simultaneous arms.

Before every arm, the harness performs the declared 60 s workload warm-up and waits until GPU temperature is within 2 °C of that pair's first-arm starting temperature. A maximum 15-minute cooldown is allowed. Failure to reach that range invalidates the complete pair. Cache state must be made equivalent by the workload's declared cache-reset procedure; if no valid reset exists, cache state is measured and pair order is included as a fixed effect in a sensitivity analysis.

The baseline records order, period, warm-up, start temperature, clock policy, throttle flags, load average, and background-process interference. A pair is invalidated if an arm reports thermal or power throttling, collector clock discontinuity, host sleep, workload/input mismatch, an undeclared background task consuming more than 2% CPU for over 5 s, or a measurement-tool failure. The complete pair is excluded; exclusions are decided from preregistered diagnostics without comparing arm outcomes.

Order, period, and carryover are tested after collection by fitting the endpoint effect to preregistered order and period indicators and by comparing the two order strata. A Holm-adjusted diagnostic p-value below 0.05 or an order-stratum estimate difference larger than one-half of the endpoint margin makes that endpoint inconclusive; it does not authorize removal of observations. Cache, thermal, or carryover evidence that violates the controls above invalidates the affected pair. If more than 10% of attempted pairs for a workload are invalid, every result for that workload is inconclusive.

## Repetitions and stopping

A separate 20-pair pilot is run for every workload. Pilot pairs never enter the final analysis. For each required workload-endpoint combination, calculate the standard deviation `s` of paired log effects or paired additive effects. Let `Δ` be the corresponding log or additive margin, `m=27`, family alpha `0.05`, and desired power `0.90`. The final independent-pair count is:

`n = max(30, max over required endpoints ceil(((z(1-0.05/(2m)) + z(0.90)) * s / Δ)^2))`.

The workload receives the maximum `n` required by its endpoints. If this exceeds 200, register 200 final pairs and classify any endpoint whose adjusted interval does not establish equivalence as inconclusive or failed as applicable. No final data are inspected before `n` is frozen. There is no early stopping, sample-size re-estimation, or optional continuation. Replacement of invalid pairs is allowed only until the frozen number of valid pairs is reached or 200 pairs have been attempted, whichever occurs first. This rule controls optional stopping by fixing sample size from disjoint pilot data.

## Uncertainty and multiplicity

For every endpoint, construct a two-sided BCa paired-bootstrap interval by resampling whole independent pairs within workload. Use 200,000 resamples and deterministic seed `8675309 + endpoint_ordinal + 1000 * workload_ordinal`. The resampling unit is the complete pair; samples within a trial are never resampled as independent observations. BCa assumes independent pairs and exchangeability within the fixed workload and environment.

Each interval has confidence `1-0.05/27 = 0.9981481481` (99.81481481%). Bonferroni coverage therefore provides at least 95% family-wise coverage across all 27 required equivalence conclusions. If BCa cannot be computed, its assumptions fail, or fewer than the frozen valid-pair count are available, the result is inconclusive. Sensitivity diagnostics never replace the preregistered primary interval.

## Measurement boundaries and authority

The external benchmark harness is authoritative for elapsed time and throughput. It starts timing immediately before releasing the target workload and stops only after the workload completes, SIA flushes buffered samples, exports data, synchronizes accelerator work, and finishes observer teardown. Deferred enabled-arm work may not be moved outside this boundary.

The harness records its own CPU, RSS, I/O, timing-call overhead, and sampling cost in both arms. A no-target calibration is run with the same sampling schedule. Calibration is reported separately and is never subtracted unless a future ADR amendment preregisters that method before data collection.

GPU timing uses explicit device synchronization at both boundaries. Device timestamps are mapped to the external monotonic clock, and maximum synchronization residual is recorded. A residual over 1 ms invalidates the pair. Buffered samples are assigned by monotonic timestamps; late arrival does not move work outside the enabled arm.

## Measurement-quality gates

Every attempted trial records expected, received, lost, and late sample counts; sample-loss rate; collector-lateness maximum and p99; clock synchronization residual; and measurement completeness. Every `gui_active` trial also records SIA GPU-activity sample count, mean, and maximum.

A trial fails quality control if sample loss exceeds 0.1%, collector lateness p99 exceeds one configured sampling interval, any required count is missing, or any required field is non-finite. A GUI trial also fails if SIA GPU-activity evidence is missing. Missing quality fields make every affected workload-endpoint result inconclusive and prevent an overall pass.

## Failures, missing data, and anomalies

Every attempted arm and pair remains in the baseline. Timeout is fixed at 120% of the workload's pilot p99 duration, rounded up to the next second before final collection. A crash, timeout, missing arm, censored value, non-finite value, or preregistered invalidation excludes the complete pair and is counted by reason. It is never imputed.

Values are not excluded merely because they are extreme. The only anomaly exclusions are impossible clock order, corrupt input identity, measurement-tool failure, or the environmental invalidations declared above. Decisions are made without viewing enabled-versus-disabled outcomes. Analysis includes all valid complete pairs.

A workload endpoint is `fail` when its complete adjusted interval is finite and lies wholly outside either margin boundary in the adverse direction. It is `pass` only when wholly inside both boundaries. All other cases, including margin crossing, insufficient pairs, missing fields, and mixed evidence, are `inconclusive`.

## Provenance and claim boundary

The baseline records source and ADR revisions, complete configuration, observer mode, host identity, OS, kernel, CPU, memory, accelerator and driver/runtime versions, firmware, power and frequency settings, isolation controls, workload/input identity, environment, and measurement-tool versions.

Development-host evidence must be labeled `development`. A target claim requires `target` evidence collected on the declared target hardware. An unmatched host, synthetic-only workload, incomplete endpoint coverage, missing quality evidence, uncalibrated estimate, or unexecuted schema cannot support a target low-observer-effect claim.

The baseline begins with every result and the overall conclusion set to `inconclusive`. A low-observer-effect claim is permitted only after the registered campaign is executed and an independent recomputation finds every gate and all 27 results passing.