# ADR-SIA-001: Observer-effect benchmark contract

- Status: Proposed
- Specification revision: `4479bcc19db72f6ad243a87e4b7271496d60d0b7`
- Requirements: R01, R12, R38, R40
- Baseline artifact: `benchmarks/observer-effect-baseline.yaml`

## Context

SIA cannot establish that it has a low observer effect from its own counters, from a single favorable run, or from an unexecuted benchmark schema. This ADR preregisters the experiment and analysis that must be completed before such a claim is permitted.

This decision implements specification section 6.4 and traces the work-item requirements as follows:

| Requirement | Contract supplied here |
|---|---|
| R01 | Low observer effect is a measured equivalence claim, not an implementation assumption. |
| R12 | Enabled and disabled runs use matched inputs, environments, timing boundaries, and paired randomized order. |
| R38 | External measurements cover target performance, accelerator behavior, SIA resource use, sample loss, and collector lateness. |
| R40 | Raw observations, provenance, exclusions, intervals, and conclusions are preserved in a reproducible baseline artifact. |

## Decision

### Claims and contrasts

The unit of analysis is one independent pair of target invocations. Inner operations or samples within an invocation describe its distribution but are not independent repetitions.

For an endpoint with positive values, observer effect is the paired log ratio

`d = log(enabled / disabled)`.

It is reported as the geometric ratio `exp(mean(d))` and percentage change. For temperature it is the paired arithmetic difference `enabled - disabled` in degrees Celsius. All margins are directionally symmetric: an unexpectedly favorable enabled result must satisfy the same equivalence test as an unfavorable result.

Four scopes are recorded independently:

1. `headless_monitor`: disabled matched wrapper versus target-local headless recording using the frozen monitor sampling plan.
2. `headless_profile`: disabled matched wrapper versus target-local headless recording using the frozen profile sampling plan.
3. `encrypted_observer`: target-local headless recording without an attached observer versus the same recording with one paired encrypted off-target observer.
4. `live_gui`: disabled matched wrapper versus local GUI collection. This is a separate characterization and cannot substitute for headless evidence.

A conclusion for one scope does not transfer to another scope, sampling plan, workload, or hardware profile.

### Required equivalence workloads

Fixture implementations and their source revision and input digests must be frozen in the baseline before pilot collection. The following definitions are normative; changing one creates a new benchmark contract.

| Workload ID | Fixed work and input | Concurrency | Measured duration and warm-up |
|---|---|---|---|
| `cpu_single` | Deterministic integer and floating-point kernels, seed 23063, fixed operation count calibrated once to at least 60 s on the target | 1 pinned worker | Three unmeasured complete invocations, then one fixed-work invocation |
| `cpu_multi` | Same kernels and seed family, fixed operation count per worker | One pinned worker per physical core; SMT state recorded and unchanged | Three unmeasured invocations, then one fixed-work invocation |
| `memory_pressure` | Deterministic seeded read/write traversal over 70% of available physical memory measured before the experiment | 1 coordinator plus one worker per physical core | Three 30 s warm-ups, then 120 s fixed-duration work |
| `disk_io` | Deterministic 8 GiB generated payload in the declared benchmark filesystem; unique file per trial | Queue depth 1 and one worker | One unmeasured file cycle, then one write, sync, read, and verify cycle; OS cache policy is unchanged and recorded |
| `gpu_compute` | Deterministic fixed-count compute kernels with input seed 23063 and verified output digest | One host feeder and one target GPU stream | Three complete warm-ups followed by at least 120 s of fixed work |
| `gpu_transfer` | Deterministic fixed byte count of bidirectional host/device transfers plus verification | One host thread and one target GPU stream | Three complete warm-ups followed by at least 120 s of fixed work |
| `gpu_thermal_power` | The `gpu_compute` fixture under the machine's pre-existing declared power/frequency policy; SIA never changes that policy | One host feeder and one target GPU stream | Warm until the preregistered thermal stratum is reached, then 600 s fixed-duration work |
| `mixed_cpu_gpu` | Deterministic fixed-count CPU preparation, GPU work, transfer, and result-validation stages, seed 23063 | One feeder plus the fixture's frozen CPU worker count and one GPU stream | Three complete pipeline warm-ups followed by at least 180 s of fixed work |

Idle-machine, long-recording, suspend/resume, and short-burst fixtures remain mandatory engineering characterizations, but they do not create a target-performance equivalence conclusion: idle has no target work, long recording tests bounded growth and loss, suspend validates invalidation, and short bursts validate coverage warnings.

### Required endpoints and margins

Both target performance endpoints are required for every equivalence workload. Accelerator endpoints are additionally required for `gpu_compute`, `gpu_transfer`, `gpu_thermal_power`, and `mixed_cpu_gpu`.

| Endpoint | Unit and per-trial aggregation | Symmetric equivalence margin | Practical rationale |
|---|---|---|---|
| `target_elapsed` | seconds; authoritative external monotonic time around the complete fixed work | ratio 0.98 to 1.02 | A 2% latency change is material during repeated performance work. |
| `target_throughput` | declared work units per second; total verified work divided by authoritative elapsed time | ratio 0.98 to 1.02 | A 2% throughput change is likewise decision-relevant. |
| `gpu_sm_clock` | MHz; time-weighted mean over the measured target interval, with the raw distribution retained | ratio 0.98 to 1.02 | A persistent 2% clock shift can confound accelerator comparisons. |
| `gpu_power` | watts; time-weighted mean over the measured target interval, with the raw distribution retained | ratio 0.97 to 1.03 | Three percent accommodates sensor granularity while detecting material power displacement. |
| `gpu_temperature` | degrees Celsius; time-weighted mean over the measured target interval, with the raw distribution retained | difference -2.0 to +2.0 C | A 2 C shift is large enough to alter boost or throttling interpretation. |

Unavailable required accelerator endpoints make that workload and the overall target-hardware conclusion inconclusive. They are not silently marked inapplicable after data inspection.

The baseline also records, for every attempted trial, SIA CPU user/system time in seconds, peak and time-weighted RSS in bytes, read/write I/O bytes, sample expected/recorded/lost counts and loss rate, maximum and p99 collector lateness in nanoseconds, and measurement-tool errors. Every GUI-active trial additionally records SIA GPU engine time or utilization and VRAM use. Missing quality fields make every affected workload-endpoint conclusion inconclusive.

### Pairing, order, and environment control

The two arms in a pair use the same executable and input digests, seed, work count, working directory, environment allowlist, affinity, priority, device, filesystem, concurrency, launch wrapper, warm-up policy, power policy, and measurement boundaries. The disabled wrapper performs the same setup and validation but does not initialize an observation mechanism.

Adjacent pairs form balanced AB/BA blocks. Block order is generated before execution by SHA-256 over the specification SHA, frozen target fingerprint, scope, workload ID, and block number. The schedule is saved before outcomes are visible. No rerandomization is allowed.

Each arm receives the fixed warm-up above. CPU affinity and order balancing address period effects. Cache policy is fixed per workload and never changed between arms. Starting CPU package temperature must differ by no more than 3 C and starting target GPU temperature by no more than 2 C within a pair. Ambient temperature, governor, power limits, frequency policy, thermal/throttle state, other-user CPU time, disk traffic, and network traffic are recorded.

For ordinary workloads, suspend/resume, device reset, topology-changing hotplug, a new throttle reason, more than five percentage points of non-target/non-SIA CPU use for over 10% of either arm, or unrelated disk bytes exceeding 1% of the fixture's bytes invalidates the pair. `gpu_thermal_power` instead uses the frozen throttle-state stratum; arms in different strata invalidate the pair. Time trend, arm order, and period are reported as diagnostics. A significant order-by-arm interaction or monotonic thermal drift makes the workload inconclusive unless the effect was covered by a preregistered stratum. Carryover is tested by comparing first-period effects with second-period effects and by the order interaction; it is never removed by inspecting favorable outcomes.

### Measurement boundaries

The external harness is authoritative. Its elapsed interval begins before observer startup and target launch and ends only after target result validation, accelerator synchronization, observer shutdown, buffer flush, export/finalization work, recorder fsync, and observer teardown complete. Deferred work therefore remains in the enabled arm. The disabled arm passes through the same wrapper and synchronization calls.

GPU completion is explicitly synchronized before the endpoint interval ends. Target and observer clocks are anchored, but target monotonic time is authoritative. Harness timing and sampling costs are calibrated in separate no-op trials, recorded, and never subtracted from only one arm. Harness processes run off the target's pinned CPUs when the target permits; the placement is identical in both arms.

### Failures, missing data, and exclusions

Every attempted arm receives an immutable attempt ID and remains accounted for. A pair is excluded only for criteria declared here and evaluated without comparing arm outcomes:

- target or harness failure, timeout, signal, output-digest mismatch, or incomplete fixed work;
- configuration, executable, input, environment, or device identity mismatch;
- suspend/resume, device reset, hotplug, background-interference, thermal, or carryover invalidation defined above;
- corrupt timestamps or loss of the external authoritative endpoint.

If either arm is excluded, the pair is excluded. Timeouts are retained as censored attempts and cause the endpoint to be inconclusive unless a separate censoring model was registered before the pilot; this contract registers no such model. Missing required SIA quality or accelerator evidence is not an exclusion: it makes the affected result inconclusive. Statistical outliers and surprising favorable or unfavorable values are retained. No exclusion may be introduced after arm outcomes are inspected.

### Repetitions and precision

A pilot of exactly 12 independent pairs per workload, scope, and hardware profile estimates paired standard deviations. Pilot pairs are excluded from final inference and are labeled `pilot` permanently.

For each required conclusion, the target simultaneous-confidence half-width is half its equivalence half-margin: 0.01 on the log-ratio scale for elapsed, throughput, and clock; 0.015 for power; and 1 C for temperature. Let K be the number of required conclusions in the scope and let s be the pilot paired standard deviation. The fixed final pair count is

`N = max(30, max_endpoint ceil((z(1 - 0.05/(2K)) * s / precision_target)^2))`, capped at 200.

The maximum across endpoints fixes one N for a workload. The calculation and pilot values are stored before final collection. If the calculated N exceeds 200, 200 pairs may be run, but failure to attain the precision target is inconclusive. There is no outcome-dependent stopping, sample-size re-estimation, or reuse of pilot observations. Consequently optional stopping does not inflate type-I error.

### Uncertainty and equivalence

Final inference uses a paired, studentized max-|t| bootstrap with 9,999 resamples. The resampling unit is the complete independent pair, resampled within workload; all required endpoint statistics are recomputed for each resample. Resample seed is SHA-256 of the specification SHA, frozen baseline revision, scope, and literal `bootstrap-v1`. The maximum absolute studentized statistic across every required workload-endpoint conclusion in a scope supplies simultaneous two-sided intervals with at least 95% family-wise coverage. The method assumes independent pairs and exchangeability within each frozen workload; order and period diagnostics must not contradict that assumption.

Equivalence is established only when the complete adjusted interval lies strictly inside both predeclared bounds. Touching or overlapping a margin, failing to reject a difference test, or observing a favorable point estimate is insufficient. An interval outside or touching a margin is `fail` when precision is adequate; missing evidence, violated assumptions, excessive required sample size, or inadequate precision is `inconclusive`.

A scope receives an overall `pass` only if every required workload-endpoint result passes. Any failed, missing, or inconclusive result prevents an overall pass. Conflicting results must be shown together and cannot be selectively summarized.

### Baseline records and reproducibility

The baseline artifact is the preregistration and result index. Before pilot execution it must replace all pending provenance fields with source revision, ADR revision, fixture/config and observer-mode revisions, host OS/kernel, CPU and memory, accelerator model, driver/runtime, power/frequency settings, environment controls, workload/input identities and digests, and every measurement-tool name/version.

Each attempted trial must be stored as an arm-labeled row or as a losslessly reproducible content-addressed reference. Required fields include scope, profile, workload, endpoint, pair, period, AB/BA order, timestamps, values and units, inclusion status/reason, target distributions, all SIA resource fields, sample-loss and collector-lateness fields, accelerator evidence, and GUI SIA GPU activity where applicable. Summaries alone are forbidden.

Development-host evidence is labeled `development_only`. It cannot support a target claim unless hardware, OS/kernel, runtime, configuration, inputs, and complete endpoint coverage match the declared target profile. Synthetic fixtures, uncalibrated estimates, or incomplete coverage cannot support a low-observer-effect claim.

## Consequences

This contract favors falsifiability over an easy pass and can legitimately produce an inconclusive result. It costs more repetitions than an unpaired benchmark, but fixes margins, exclusions, stopping, multiplicity, and reporting before implementation outcomes are known.

The status of this ADR and its baseline remains Proposed. Neither their existence nor an unexecuted schema claims low observer effect. A claim is permitted only after this exact procedure runs on the declared target, the artifacts receive revisions, and every gate for the claimed scope passes.