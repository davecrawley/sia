# SIA - System Information Analyzer

## Performance Profiler Extension Specification

**Canonical specification.** Revision history is maintained by Git. This Markdown specification was adopted from the final committee-reviewed Version 10 dated 27 August 2026.

*Design intent: preserve SIA's clarity and simplicity while extending it into a low-overhead whole-system performance profiler suitable for IRIS and other CPU/GPU workloads.*

| Document item              | Value                                                                                                                                                                                                                                    |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Repository baseline        | davecrawley/sia, main, reviewed at commit 524ecd52746419131b5209364f327c09d0ced114                                                                                                                                                       |
| Current released package   | SIA 0.0.1                                                                                                                                                                                                                                |
| Target milestone           | SIA 0.1.0 - low-overhead performance profiler, IRIS-ready                                                                                                                                                                                |
| Primary platform           | Linux; Debian/Ubuntu packaging retained                                                                                                                                                                                                  |
| Primary accelerator target | NVIDIA GPU on the primary Intel-CPU/NVIDIA-GPU development machine. AMD/Intel GPU providers may be implemented capability-first but remain hardware-unverified until tested on real hardware.                                            |
| Primary consumer           | General-purpose local workloads; IRIS is the first concrete profiling customer                                                                                                                                                           |
| Status                     | Final committee reviewed. Human-readable three-word PAKE pairing, encrypted off-target observer, test-ownership split and hardware-verification boundaries have converged; implementation parameters remain gated by Phase 0 ADRs/tests. |

# 1. Executive summary

SIA is currently a compact Rust/egui real-time system monitor that shows CPU, RAM, GPU and VRAM utilization, temperatures and CPU/GPU clocks. The repository is deliberately small: the current implementation is essentially one src/main.rs, with Linux hwmon/cpufreq collection, sysinfo for CPU/RAM, and optional NVIDIA NVML support. The README already asks for lower overhead, headless operation, CSV recording and application-code triggering. This specification turns those wishes into a coherent profiler architecture without turning SIA into a replacement for Nsight, perf, VTune or a distributed observability platform.

The central architectural change is separation of collection, recording and presentation. A headless collector becomes the authoritative measurement path; the GUI becomes one consumer of live or recorded sessions. This is particularly important for GPU profiling because the current egui/eframe build uses wgpu and requests a repaint every 16 ms even though samples are taken at 1 Hz. The profiler must be able to measure GPU workloads without its own GUI activity materially perturbing the GPU.

SIA 0.1.0 should answer questions such as: Which stage of an application was running when the machine slowed down? Was the accelerator waiting for a CPU feeder? Did power or thermal throttling reduce clocks? Was the workload limited by I/O or memory pressure? Was VRAM close to capacity? Did a change make end-to-end execution faster, or merely increase a utilization percentage? SIA should present evidence and plausible bottleneck hypotheses, then direct the user to deeper specialist tools when kernel- or instruction-level analysis is required.

## 1.1 Committee conclusion

The second hostile pass found no reason to change the basic three-word PAKE architecture, but it removed two unnecessary commitments and closed several stale contracts. SIA 0.1 does not need local phrase regeneration while a session is running: the phrase lasts for that headless session, and restarting headless is the simple recovery if the local credential is believed compromised. The specification also stops pretending a particular Rust SPAKE2 crate has already been security-approved; the security ADR must choose a maintained implementation of a published PAKE with the required offline-guess resistance and key confirmation, using SPAKE2/RFC 9382 as the reference balanced-PAKE construction rather than a preselected dependency. Stale SSH and plaintext-debug language has been removed. The committee's final position is:

- Keep one SIA executable and one codebase; use clear modules rather than introducing a daemon/microservice architecture.

- Make `sia -headless` the authoritative performance-measurement mode. It MUST NOT initialize eframe/wgpu/OpenGL/Vulkan or create a graphics device/window. It records locally and can expose a bounded encrypted observer stream; benchmark authority remains the target-local recording.

- Use an explicit clock-domain contract, not merely a 'monotonic timestamp': native samples use Linux CLOCK_MONOTONIC nanoseconds plus boot/time-namespace identity; imported clocks require a declared domain or synchronization anchors. CLOCK_BOOTTIME anchors detect suspend/resume, which invalidates ordinary benchmark comparisons unless explicitly allowed.

- Add a capability-driven metric model: unsupported metrics are absent or explicitly unavailable, never silently rendered as zeros or dead legend entries.

- Record transparent append-only session streams with compact wide CSV rows per collector/entity family plus JSONL events/metadata; export a canonical long-form CSV/JSONL view. The recorder is bounded and nonblocking, detects dropped frames and remains readable after crashes/disk exhaustion.

- Extend NVIDIA telemetry substantially with NVML and prefer timestamped driver-maintained sample buffers/counters where available, because they can provide higher-frequency data with lower polling cost. Polling remains the fallback. Do not reimplement Nsight/CUPTI. AMD/Intel use stable kernel/vendor capabilities and DRM client stats where available.

- Make SIA's own observer effect visible and test it systematically before using SIA to optimize IRIS.

- Make the synchronized evidence timeline the product. Automatic bottleneck annotations are a later transparent convenience and are explicitly outside the SIA 0.1 IRIS-ready acceptance gate.

- Define an IRIS trace importer and general application-marker protocol, but do not create an IRIS-specific SIA product.

- Treat visual scales/labels as measurement semantics: no hidden rescaling, no mixed-unit/dual-axis comparisons, no rainbow-as-default, and no line interpolation through missing data.

- Use progressive disclosure in the GUI: overview -> synchronized focus/zoom -> details/evidence. Do not put every core, sensor, engine and diagnostic on the opening screen.

- Schedule sampling from absolute monotonic deadlines, record lateness and skip missed ticks rather than burst-catching up; actual observation time always remains authoritative.

- Headless capture must be genuinely headless: no eframe renderer, graphics context/device, plot construction or GUI timer is initialized on the `sia -headless` path. The observer computer performs all GUI rendering.

## 1.2 IRIS readiness

IRIS v21 requires low-overhead performance spans keyed by RunId/stage/workload, monotonic start/end timestamps, CPU/GPU/backend identity, work-unit/batch counts, hardware/runtime profile identity and numerical/result status, with optional device-memory and transfer metadata. The primary Intel-CPU/NVIDIA-GPU development machine must support authoritative target-local `sia -headless` recording and live off-target observation through `sia -o <IP> <three-word-phrase>`. For reliable offline alignment, imported IRIS traces declare the actual clock domain and boot/time-namespace identity or synchronization anchors. AMD/Intel-GPU providers may be implemented against documented interfaces and fixtures but cannot be labeled hardware-verified until real hardware is available.

# 2. Current SIA baseline

## 2.1 Repository architecture

| Area          | Current implementation                             | Profiler consequence                                                                |
|---------------|----------------------------------------------------|-------------------------------------------------------------------------------------|
| Application   | Single Rust binary; almost all code in src/main.rs | Refactor by module before adding major new collection/storage/UI paths.             |
| UI            | eframe/egui + egui_plot, wgpu feature              | Convenient and attractive, but live GPU rendering can create observer effect.       |
| CPU/RAM       | sysinfo refresh_cpu/refresh_memory                 | Useful baseline; add per-process and pressure/I/O context.                          |
| Thermals      | Linux /sys/class/hwmon discovery                   | Good general Linux source; preserve capability-driven discovery.                    |
| CPU frequency | Per-core cpufreq sysfs reads                       | Useful; collection implementation should avoid needless reopen/allocation overhead. |
| NVIDIA        | nvml-wrapper 0.11.0; GPU 0 only                    | Provider should enumerate all GPUs and expose a capability set.                     |
| History       | In-memory RollingSeries VecDeque                   | Good live buffer; insufficient as authoritative recorded session.                   |
| Sampling      | App::new(5\*60, 1.0); nominal seconds increment    | Timestamp samples from the clock; history capacity must be time/rate aware.         |
| Rendering     | request_repaint_after(16 ms)                       | Potentially ~60 redraws/sec for 1-Hz data; must become event/sample driven.         |

## 2.2 Specific current defects relevant to profiling

1.  Observer effect: the GUI uses a GPU-capable renderer and requests 16-ms repaints even when no new metric sample exists. SIA can therefore create load on a GPU it is trying to measure.

2.  Nominal timebase: App::sample increments `seconds` by the requested sample period rather than recording actual sample time. Sampling jitter or stalls therefore become invisible and application-event correlation can drift.

3.  Per-frame allocation/work: RollingSeries::points_after and points_after_scaled create new vectors when the plot is redrawn. With a 16-ms repaint schedule this repeats work for unchanged samples.

4.  Collector/UI coupling: sampling, data ownership and rendering live in the App object. Headless recording, replay and testability require a collector/session boundary.

5.  GPU capability ambiguity: current code queries both `util.gpu` and `util.memory` but keeps only GPU utilization; `VRAM %` is instead occupancy derived from used/total bytes. Memory-controller activity and VRAM occupancy are distinct metrics and must remain distinct.

6.  Single NVIDIA device: device index 0 is hard-coded. Multi-GPU systems are not represented correctly.

7.  Unsupported-device UI: the NVIDIA feature path can create GPU traces/legend concepts even when useful NVIDIA metrics are unavailable, matching the existing issue that dead GPU lines remain visible on non-supported systems.

8.  Portability gap: hwmon classification recognizes AMD GPU temperatures, but no AMD GPU utilization/VRAM backend exists. Existing AMD issue #7 is therefore architectural rather than a plotting bug.

9.  Fixed sample-count history: the RollingSeries capacity is passed as `capacity_secs`, but it is actually a count of samples. Increasing sample rate silently reduces the retained time window.

10. No recorded causality: there is no process attribution, application span/event stream, persistent session, or replay mode, so a utilization spike cannot be tied reliably to the code that caused it.

## 2.3 Existing repository intent and issues

| Repository item | Current request/problem                      | Specification response                                                                                                                                                                                                                     |
|-----------------|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| README roadmap  | Performance improvements                     | Phase 0/1 observer-effect baseline and low-overhead refactor.                                                                                                                                                                              |
| README roadmap  | Headless operation / remote monitoring       | Headless collector/recorder is core. A SIA GUI on another computer connects directly to the target headless encrypted read-only stream using the exact printed `sia -o <IP> <phrase>` command; browser/web viewing remains deferred. |
| README roadmap  | CSV triggering / later analysis              | Native recorded sessions + CSV export.                                                                                                                                                                                                     |
| README roadmap  | Triggering inside code                       | Generic application span/marker schema + optional Unix socket/CLI + offline trace import.                                                                                                                                                  |
| Issue #1       | SIA consumes too many resources              | Explicit self-overhead benchmarks and sample-driven rendering.                                                                                                                                                                             |
| Issue #3       | GPU graphs nonfunctional but still in legend | Capability-driven providers; unavailable metrics are omitted.                                                                                                                                                                              |
| Issue #4/#5    | Small-screen/window/column problems          | Responsive GUI refactor; remove hard oversized minimum layout assumptions.                                                                                                                                                                 |
| Issue #6       | Need headless mode                           | Core 0.1 requirement.                                                                                                                                                                                                                      |
| Issue #7       | AMD GPU utilization/VRAM absent              | AMDGPU provider phase with stable sysfs/hwmon capabilities.                                                                                                                                                                                |

# 3. Scope

## 3.1 Goals

- Preserve SIA's original product idea: a clean, legible view of what is hot, busy, constrained or idle without drowning the user in redundant counters.

- Measure whole-system behavior with sufficiently low and measurable overhead that SIA can be trusted during performance work.

- Record sessions headlessly, then inspect them offline with the GUI.

- Correlate system metrics with application spans/events and target processes.

- Expose enough GPU telemetry to distinguish broad classes of under-utilization, capacity pressure and throttling.

- Support CPU, memory, I/O and Linux pressure-stall context so GPU under-utilization can be explained by upstream bottlenecks.

- Support multiple devices and partial-capability hardware without fake traces.

- Provide general-purpose import/export and marker semantics; IRIS is a first customer, not a hard-coded special case.

- Make measurements reproducible: session manifest, hardware/software fingerprint, clock model, sampling plan, dropped-sample/event counts and SIA version are all recorded.

- Remain a small Rust application with a comprehensible architecture and conservative dependency set.

## 3.2 Non-goals for SIA 0.1

- Replacing NVIDIA Nsight Systems/Compute, Linux perf, Intel VTune, ROCm profiling tools or kernel tracing frameworks.

- Collecting CUDA warp-stall, tensor-core, cache-line or instruction-level hardware counters directly.

- Automatically changing the profiled application's CPU affinity, GPU clocks, power limits, batch size or code.

- A distributed multi-host observability backend, cloud service or permanent telemetry daemon.

- A database server. Recorded sessions are local files and can be archived or exported.

- Machine-learned bottleneck classification. v0.1 diagnostics are transparent rules with visible evidence.

- Windows/macOS feature parity. Linux is the normative platform for the IRIS milestone.

- Requiring target applications to link a SIA-specific library.

## 3.3 Normative terms

MUST/MUST NOT are release requirements. SHOULD/SHOULD NOT are strong defaults that may be changed only with a documented reason. MAY denotes an optional capability. 'Available' means the provider can obtain the metric reliably on the current device and permission context; absence is not treated as zero.

# 4. Architecture

## 4.1 Component model

SIA remains one product and one executable. The implementation should be split into modules with narrow interfaces rather than separate services. The GUI, headless recorder and offline viewer all consume the same typed session/metric model.

```text
src/  
main.rs  
cli.rs  
clock.rs  
model.rs  
collector/  
mod.rs  
cpu.rs  
memory.rs  
pressure.rs  
disk.rs  
network.rs  
hwmon.rs  
process.rs  
gpu/  
mod.rs  
nvidia.rs  
amd.rs  
intel.rs  
session/  
mod.rs  
writer.rs  
reader.rs  
export.rs  
events/  
mod.rs  
schema.rs  
unix.rs  
import_iris.rs  
gui/  
mod.rs  
overview.rs  
timeline.rs  
processes.rs  
diagnostics.rs  
diagnostics.rs
```

The exact file split is non-normative; the module boundaries are normative. A future reason to split crates must be justified by build/test/reuse needs rather than architecture fashion.

## 4.2 Runtime data flow

```text
hardware/provider discovery  
\|  
v  
SamplingPlan  
\|  
v  
Collector thread(s) ----> bounded sample channel ----> SessionWriter (optional)  
\| \|  
\| +---------------> live GUI  
\|  
+----> marker/event receiver -----------> SessionWriter / live GUI  
\|  
+----> process/GPU provider capability state  
  
offline: recorded session + imported application trace ---> SessionReader ---> GUI / export / diagnostics
```

Sampling MUST continue independently of GUI frame rate or observer-client state. The scheduler uses absolute monotonic deadlines per collector rather than `sleep(period)` after work, so collection cost does not accumulate into schedule drift. When late, it records lateness and skips missed deadlines instead of issuing catch-up bursts. A slow provider MUST NOT indefinitely delay unrelated collectors. Local recording is authoritative; observer streams are secondary consumers and may lose optional display updates without changing target collection timestamps.

## 4.3 Headless target and off-target observer GUI

`sia` starts the local live GUI; `sia -headless` (also accepted as `--headless`) starts authoritative headless collection/recording plus the minimal read-only observer transport; `sia -o <target-ip> <pairing-phrase>` (also `--observer`) starts the GUI on another computer; and `sia view <session>` opens a recording. The measured machine performs no GUI/rendering work. A background daemon, account service or browser server is not required for SIA 0.1.

### 4.3.1 Observer source and operator contract

The GUI consumes a SessionSource abstraction with three equivalent presentation sources: LocalLiveSource, ObserverLiveSource and RecordedSessionSource. All three expose the same MetricDescriptor/Sample/Event/Status model, and the GUI must not contain separate rendering logic for observer data. Target identity is part of the source and is always displayed.

The normal operator path is intentionally copyable. After `./sia -headless`, SIA enumerates suitable non-loopback LAN addresses and prints the exact observer command for each address, for example `./sia -o 192.168.1.42 big-pink-elephant`. The observer uses a fixed default SIA port chosen by the implementation ADR, so a port need not appear in the ordinary command. If the default port is unavailable or an override is requested, headless prints `IP:port` in the exact command. The versioned framed transport carries session/capability metadata, metric descriptors, samples, events, data-quality records and terminal status.

Observer streaming MUST be bounded and nonblocking with respect to authoritative collection and local recording. If the network or observer GUI cannot keep up, the stream may coalesce/drop optional display samples while preserving ordering/sequence/loss accounting; it may never stall the collector, recorder or profiled application. Target monotonic timestamps and target identity remain authoritative; observer-wall-clock timestamps are presentation metadata only.

Observer live viewing is a first-class diagnosis/operations feature but not automatically a benchmark-authoritative mode. Serialization, encryption and network transmission consume target resources. Performance claims use target-local headless recording with no observer attached unless a separately registered repeated/interleaved equivalence experiment demonstrates that `headless + observer` is decision-neutral for that workload/hardware profile.

SIA 0.1 requires one observer client; support for more than one is optional and benchmark-derived. The protocol is read-only after authenticated setup: the observer cannot change clocks, sampling plans, process priority, GPU state or the profiled application. Browser/web viewing and remote control remain deferred.

### 4.3.2 Human-readable pairing and encrypted channel

On every headless start SIA generates a fresh three-word pairing phrase from the operating-system CSPRNG and versioned curated word lists, using a memorable `adjective-adjective-noun` form such as `big-pink-elephant`. The phrase is session-scoped, lower-case ASCII after canonicalization, printed locally, never written to the recorded session, and accepted as one hyphenated command-line token. Phrase-space size is not a magic constant: the security ADR derives the required number of combinations from the registered LAN threat model, maximum online attempts/backoff policy and required upper bound on per-session online-guess success, then chooses the largest human-friendly curated lists needed to exceed that bound. Word-list quality is separately human-reviewed for spelling, ambiguity, offensiveness and memorability.

The three-word phrase MUST NOT be used as a raw encryption key, directly hashed into a long-term key, or exposed in cleartext on the network. It is the low-entropy shared secret for a standard password-authenticated key exchange that resists passive/offline dictionary attack and derives a strong ephemeral shared secret. The reference protocol is symmetric SPAKE2 as specified in RFC 9382, or a security-reviewed equivalent PAKE selected by ADR; no hand-rolled PAKE is permitted. After explicit key confirmation, HKDF (or the PAKE library's equivalent schedule) derives directional authenticated-encryption keys for the telemetry stream. The cipher suite, protocol/library version and transcript-binding rules are versioned in the protocol manifest.

The headless listener binds only to explicitly selected local interfaces or the configured safe default. It prints every listening address/port. Command-line options allow `--listen <addr>` and `--port <port>`. There is no remote mutation/control API in 0.1.

### 4.3.2.1 Normative headless output

At successful startup the terminal MUST contain a compact copy/paste block equivalent to:

```text
SIA headless profiler running  
Recording locally: ./sia-sessions/<session>/  
Observed machine: <host>  
Observer address: 192.168.1.42  
Pairing phrase: big-pink-elephant  
On the observer computer type:  
./sia -o 192.168.1.42 big-pink-elephant  
```

If more than one plausible LAN address exists, print one complete command per address. If a non-default port is necessary, print `IP:port`. The operator must not need to infer a port, concatenate a credential, or consult documentation.

The exact wording may evolve with information-design review, but the semantic fields and complete copyable command are normative.

### 4.3.3 Pairing attempts, failure and network exposure

A failed pairing attempt reveals only failure. Online guesses are rate-limited and bounded by a policy derived in the security ADR; collection and local recording continue normally during backoff. One active observer is sufficient for SIA 0.1. While one observer is connected, additional observer attempts are rejected without disturbing the active stream. If the observer disconnects, the same session phrase remains valid for reconnect until the headless session ends. If the operator believes the phrase is compromised, the simple 0.1 recovery is to stop/restart the headless SIA session rather than add a remote credential-management control path. The PAKE must make captured traffic unusable for offline phrase guessing.

The default `sia -o <IP> <phrase>` form deliberately favors human usability. The phrase can therefore appear transiently in the observer machine's process arguments and may be retained by that user's shell history. This is acceptable only because the credential is ephemeral and dies with the headless session, and because local users on the observer machine are inside the 0.1 trust boundary. SIA should overwrite/redact its own argv copy where the platform permits, but cannot reliably control shell history. An optional `sia -o <IP>` prompt mode MAY be provided for users who do not want the phrase in shell history; the headless machine still prints the simple full command by default.

The 0.1 threat model is a trusted or semi-trusted local LAN: resist passive sniffing, casual unauthorized attachment and active man-in-the-middle attempts that do not know the displayed phrase. Internet exposure is unsupported. By default, headless ranks plausible private/link-local LAN interfaces, excludes loopback and obviously virtual/container/tunnel interfaces from the recommended copy/paste list unless explicitly requested, binds only to the selected safe interface set, and prints every listening address. `--listen`/`--port` override discovery. IPv6 commands use unambiguous bracketed address syntax. A future Internet-capable remote mode requires a separate security review.

Pairing and bulk encryption are measured as observer load. The external observer-effect suite compares workload alone; workload + headless local recording; and workload + headless local recording + paired encrypted observer. Encryption is neither assumed free nor allowed to degrade the authoritative recorder path.

The implementation MUST use a maintained cryptographic library and published protocol/test vectors. The security ADR records the selected PAKE specification/implementation, maintenance and independent-review/audit evidence, cipher suite, word-list version, online-attempt/backoff policy and benchmarked target-side cost. SPAKE2/RFC 9382 is the reference balanced-PAKE construction because both peers know the ephemeral phrase, but it is not a mandate to use an unaudited crate; a different established PAKE may be selected if its implementation assurance is materially stronger and the user interaction remains identical. Custom cryptographic primitives or home-grown password-to-key schemes are prohibited.

## 4.4 Observer-effect rule

Any SIA measurement used to choose an IRIS CPU/GPU implementation or claim a speedup MUST use target-local `sia -headless` recording with no observer attached unless the encrypted-observer path has separately passed the registered equivalence test for that workload. Observer live GUI remains a core operational feature. Every session records SIA's own process CPU/RSS/I/O metrics and capture mode (`headless_no_observer`, `headless_encrypted_observer`, `live_local`) so observer cost is explicit.

# 5. Time, identity and data model

## 5.1 Clock contract

Native SIA samples use `clock_gettime(CLOCK_MONOTONIC)` and record nanoseconds in an explicitly named `linux_clock_monotonic` domain. The session records Linux boot ID and time-namespace identity. This makes timestamps comparable across ordinary processes on the same boot only when they inhabit compatible time namespaces. SIA also records CLOCK_BOOTTIME/UTC synchronization anchors at session start/end and periodically in long sessions so suspend/resume or clock-domain translation can be detected. The authoritative sample timestamp is the actual observation time, never a synthetic counter advanced by the configured sample period.

- Sampling jitter is observable: actual observation timestamps and, for interval-derived metrics, the observation window are retained.

- Wall-clock/NTP changes do not reorder native samples because session ordering uses CLOCK_MONOTONIC; UTC exists only for human correlation/export.

- Imported events MUST declare `clock_domain`. A source on another monotonic/custom clock needs one or more synchronization anchors; otherwise SIA may align only approximately and MUST label the quality.

- A live event received without a sender timestamp is stamped on receipt and marked `arrival_timestamped`; transport latency is therefore part of its alignment uncertainty.

### 5.1.1 Suspend/resume and benchmark validity

CLOCK_MONOTONIC intentionally excludes time spent suspended, while CLOCK_BOOTTIME includes it. SIA compares these anchors to detect suspend/resume. A benchmark session containing suspend/resume, major device reset or topology-changing hotplug is marked `environment_changed` and is invalid for ordinary before/after speed claims unless the benchmark protocol explicitly studies that event. The captured session remains viewable.

## 5.2 Stable identities

| Entity           | Stable identity requirement                                                            | Notes                                                                                                                             |
|------------------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| Session          | UUID + start boot_id/monotonic anchor                                                  | One capture, live or headless. Remote sessions also record target host fingerprint/boot identity separately from viewer identity. |
| GPU              | Vendor UUID when available; otherwise PCI domain:bus:device.function + driver identity | Never 'GPU 0' as durable identity.                                                                                                |
| CPU core         | Linux logical CPU id + topology metadata                                               | Topology can group SMT siblings/package.                                                                                          |
| Process          | PID + `/proc/<pid>/stat` start time; pidfd retained where Linux supports it        | PID/start-time is durable session identity; pidfd provides race-resistant lifecycle monitoring for attached/launched roots.       |
| Thread           | TID + process identity + start observation                                             | Optional high-detail collection.                                                                                                  |
| Metric           | Versioned string metric_id + descriptor                                                | Units/meaning live in descriptor, not plot label.                                                                                 |
| Application span | source + trace_id + span_id                                                            | Parent id optional.                                                                                                               |

## 5.3 Metric descriptor

```text
MetricDescriptor {  
metric_id: string  
display_name: string  
entity_kind: system\|cpu\|gpu\|disk\|net\|process\|thread\|application  
unit: canonical unit  
value_kind: gauge\|counter\|rate\|state  
temporal_semantics: point_sample\|interval_average\|interval_delta\|cumulative_counter\|vendor_sampled  
provider: sysinfo\|procfs\|sysfs\|nvml\|drm\|imported\|...  
capability_status: available\|unsupported\|permission_denied\|temporarily_unavailable  
source_resolution_hint: optional duration  
source_semantics: short documented definition  
comparability_group: optional string  
semantics_version: integer  
}
```

A metric that is unsupported MUST NOT be emitted as 0 or displayed as a flat zero trace. `temporarily_unavailable` samples MAY be represented as gaps with a status reason. SIA MUST preserve the source's temporal semantics: a vendor utilization percentage averaged over an internal sample window is not labeled as an instantaneous load, and metrics from different vendors are not declared quantitatively interchangeable merely because both use percent units.

## 5.4 Sample record

```text
MetricSample {  
mono_ns: u64 // observation/end time in declared session clock domain  
window_start_mono_ns: optional u64  
metric_id: string  
entity_id: string  
value: f64\|i64\|u64\|state  
status: ok\|stale\|temporarily_unavailable\|error  
}
```

## 5.5 Application event/span record

```text
ApplicationEvent {  
timestamp_ns: optional u64  
clock_domain: optional string  
utc_ns: optional i128  
boot_id: optional string  
time_namespace_id: optional string  
source: string  
trace_id: string  
span_id: optional string  
parent_span_id: optional string  
pid: optional u32  
tid: optional u32  
category: string  
name: string  
kind: span_begin\|span_end\|instant\|counter  
value: optional number  
attributes: bounded small map<string, scalar>  
sequence: optional u64  
}
```

The event model intentionally mirrors the general slice/span/event/counter concepts used by Rust `tracing` and Perfetto TrackEvent without requiring either runtime. Clock identity is first-class because Perfetto-style multi-clock traces are alignable only when clock domains or synchronization snapshots are known. Attribute count, key/value length and datagram payload size are bounded to prevent an instrumented application from exhausting SIA memory.

# 6. Sampling and low-overhead design

## 6.1 SamplingPlan

SIA uses a versioned SamplingPlan rather than one hard-coded global frequency. Each collector declares metrics, temporal semantics/source resolution, estimated collection cost, cadence and scheduling phase. The scheduler derives absolute CLOCK_MONOTONIC deadlines. Slow/expensive collectors may run less often or be phase-staggered so they do not create artificial periodic load spikes; SIA does not claim such measurements are simultaneous unless their actual observation windows overlap.

Phase 0 MUST benchmark candidate cadences on the target IRIS-class machine before final defaults are frozen. Sampling defaults are therefore implementation parameters derived from measurement, not investment-system parameters and not tuned against IRIS financial outcomes.

### 6.1.1 Sampling semantics and aliasing

SIA is a sampled profiler, not a complete event tracer. A source may itself average over an undocumented or vendor-defined interval; polling it faster does not manufacture higher temporal resolution. Each metric descriptor therefore records the best known source resolution/window semantics. The UI shows sampling cadence/resolution on demand, and diagnostics MUST NOT infer that a short event did not occur merely because no sampled point captured it. For counters such as CPU time, DRM engine busy time or PSI totals, SIA SHOULD prefer interval deltas over repeatedly treating the source as an instantaneous gauge.

Provider cadences are benchmarked independently. Faster sampling is justified only when it reveals decision-useful structure at acceptable observer cost and the underlying source can actually support it. SIA may use event/poll mechanisms (for example PSI triggers) for specific conditions where the kernel exposes them, but v0.1 does not require a general event-driven kernel tracer.

### 6.1.2 Prefer source-native counters/sample buffers

When a source provides cumulative counters or a timestamped internal sample buffer, SIA SHOULD use those semantics rather than polling a coarse gauge faster. Examples include `/proc`/DRM/PSI counters and NVML's sample-buffer APIs for supported utilization/power/clock sample types. This often improves temporal fidelity while reducing observer cost. Provider capability discovery records whether SIA is reading a native sample stream, deriving an interval delta, or polling a point/vendor-sampled value.

## 6.2 GUI repaint policy

- Remove the unconditional 16-ms repaint schedule for static data.

- Request repaint when a new sample/event arrives, when an animation is active, or when the user interacts; otherwise sleep until the next expected update.

- Typography/style objects SHOULD be rebuilt only when settings change, not on every frame.

- Plot geometry for unchanged data SHOULD be cached; the GUI SHOULD NOT perform O(history_length) allocation/copy work on every idle repaint.

- Long timelines SHOULD use display decimation that preserves extrema/spikes (for example min/max envelopes per pixel bucket) rather than dropping arbitrary points.

- GUI performance itself is benchmarked and shown in the self-overhead panel.

## 6.3 Collection efficiency

- Sensor/device discovery occurs at startup and on explicit rescan/hotplug events; it is not repeated per sample.

- File descriptors or parsed topology metadata SHOULD be reused where safe rather than reopening large numbers of sysfs/procfs files on every tick.

- Providers SHOULD batch related queries when the underlying API permits it.

- Sampling threads MUST avoid busy waiting.

- No collection path may block the target application.

- Expensive optional metrics can run at a slower cadence than core utilization/pressure metrics.

- Each collector records its own collection duration, scheduling lateness and error/drop counters at a low enough cadence to diagnose SIA overhead without recursively flooding the trace.

## 6.4 Overhead acceptance methodology

Before profiler implementation proceeds beyond Phase 0, the project freezes an `ObserverEffectTest` ADR containing practical equivalence margins for headless monitor/profile modes and a separate live-GUI observer characterization. Margins are stated in terms of target-workload elapsed time/throughput and, where relevant, accelerator behavior; they are not arbitrary SIA-CPU percentages. The test design uses repeated interleaved control/SIA trials, records thermal/power/background state, and reports effect sizes with uncertainty intervals rather than one lucky timing. The number of repetitions is chosen from observed variance/desired precision following rigorous benchmarking principles rather than a fixed magic count.

External timing/benchmark harness measurements are authoritative for observer-effect claims; SIA must not certify its own neutrality solely from counters it collected itself. Control and profiled trials use the same launch wrapper/environment/working directory and differ only in whether collection is enabled, so parent-process/setup effects are not confused with measurement overhead. Results record target elapsed/throughput distribution, SIA CPU time/RSS/I/O, SIA GPU activity where any GUI is active, sample loss, collector lateness, target GPU clocks/power/temperature and machine/runtime fingerprint. Sessions with uncontrolled suspend, major background interference or thermal-state drift are retained but excluded or stratified according to the predeclared benchmark protocol.

# 7. Metric catalog

## 7.1 System and CPU

| Metric family                                        | Required/optional             | Source                                                                                                                                                                                                                                                          | Purpose                                         |
|------------------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| System + per-core CPU utilization                    | MUST                          | sysinfo or procfs                                                                                                                                                                                                                                               | Identify CPU saturation and feeder bottlenecks. |
| CPU frequency per logical/core/package where exposed | MUST where available          | cpufreq sysfs                                                                                                                                                                                                                                                   | Detect frequency collapse/boost behavior.       |
| CPU/package temperatures                             | MUST where available          | hwmon                                                                                                                                                                                                                                                           | Thermal context.                                |
| Load/run-queue context                               | SHOULD                        | Use load averages plus runnable-task indicators such as `procs_running`; distinguish queued demand from utilization.                                                                                                                                          | Separate CPU demand from low utilization.       |
| Pressure Stall Information: CPU/memory/I/O           | MUST where kernel exposes PSI | Use `/proc/pressure/\*` totals as cumulative counters and derive interval stall share from deltas; avg10/60/300 are context only. CPU `some` is the principal system CPU-pressure signal; interpret provider/kernel-specific `full` semantics cautiously. | Direct system-stress/stall signal.              |
| Context switches / interrupts                        | SHOULD                        | /proc/stat                                                                                                                                                                                                                                                      | Diagnose scheduling/interrupt pressure.         |
| CPU package energy/power                             | MAY                           | RAPL sysfs                                                                                                                                                                                                                                                      | Power/thermal explanation.                      |
| Thermal throttle counters/reasons                    | MAY where stable ABI exists   | sysfs/vendor                                                                                                                                                                                                                                                    | Direct throttle evidence.                       |

## 7.2 Memory

| Metric                       | Requirement          | Purpose                                                          |
|------------------------------|----------------------|------------------------------------------------------------------|
| RAM used/available/total     | MUST                 | Capacity and pressure context.                                   |
| Swap used + swap-in/out rate | SHOULD               | Detect memory overcommit effects.                                |
| Major/minor page faults      | SHOULD               | Correlate stalls with paging/fault activity.                     |
| Memory PSI                   | MUST where available | Detect actual memory stall pressure rather than occupancy alone. |

## 7.3 Storage and network

| Metric                                        | Requirement          | Source/notes                                          |
|-----------------------------------------------|----------------------|-------------------------------------------------------|
| Per-device read/write bytes/s                 | SHOULD               | /proc/diskstats or stable equivalent.                 |
| I/O operations + device busy/queue indicators | SHOULD               | Enough to distinguish ingest/storage stalls.          |
| I/O PSI                                       | MUST where available | /proc/pressure/io.                                    |
| Per-target-process read/write bytes/s         | MUST in process mode | /proc/<pid>/io where permitted.                     |
| Per-interface RX/TX bytes/s                   | SHOULD               | Useful for remote data ingestion and model downloads. |

## 7.4 Common GPU semantic model

GPU providers expose a capability set. The common UI uses stable semantic concepts only where their meanings genuinely overlap; each provider retains a precise source definition and comparability group. A 70% NVIDIA 'kernel busy during the vendor sample period', a 70% AMD aggregate busy metric and a 70% DRM-engine interval utilization are not silently treated as the same physical quantity. Cross-vendor side-by-side comparison is descriptive unless the metric descriptors explicitly declare compatible semantics.

| Common metric                                 | Requirement                   | Interpretation caution                                                                                     |
|-----------------------------------------------|-------------------------------|------------------------------------------------------------------------------------------------------------|
| GPU device identity / PCI / driver            | MUST                          | Required to reproduce a session.                                                                           |
| Compute/graphics utilization                  | MUST if provider supports     | Time busy is not the same as fraction of peak FLOPS.                                                       |
| Memory-controller / memory-access utilization | SHOULD if provider supports   | Distinct from VRAM occupancy and not a direct bandwidth-saturation percentage unless vendor defines it so. |
| VRAM used/free/total                          | MUST if discrete VRAM exists  | Capacity occupancy, not bandwidth.                                                                         |
| Temperature                                   | MUST where supported          | Combine with throttle evidence.                                                                            |
| SM/graphics + memory clocks                   | SHOULD                        | Explain idle/power/throttle states.                                                                        |
| Power usage + enforced/board limit            | SHOULD                        | Evidence for power capping and energy use.                                                                 |
| Performance/power state                       | SHOULD                        | Vendor-specific meaning shown in details.                                                                  |
| Throttle/clock event reasons                  | SHOULD                        | Strong direct diagnostic evidence.                                                                         |
| PCIe link width/speed + RX/TX throughput      | SHOULD where available        | Useful for transfer-heavy workloads; sampling interval must be recorded.                                   |
| Fan / memory temperature                      | MAY                           | Hardware-dependent.                                                                                        |
| Per-process GPU memory/utilization            | SHOULD where provider permits | Correlate target process with system GPU activity.                                                         |

### 7.4.1 Device/provider lifecycle

Devices/providers have explicit lifecycle state: discovered, available, temporarily_unavailable, reset/lost, removed and reappeared. A failed read does not freeze the last value into the future. GPU reset/fall-off-bus/hotplug creates a visible event/gap; if the same stable device returns it begins a new provider generation. A topology-changing event marks ordinary benchmark comparisons `environment_changed` while preserving the trace.

## 7.5 NVIDIA provider

The existing `nvml-wrapper` integration is retained but generalized from a single hard-coded device to device enumeration by stable UUID/PCI identity. The current wrapper/API already exposes utilization rates, VRAM, process utilization samples, power usage, PCIe throughput, clocks, temperature and throttle reasons on supported hardware. Calls that return NotSupported are capability results, not application failures.

- Store both NVML GPU utilization and NVML memory utilization. Label them according to NVML semantics: percent of the vendor sample period with one or more kernels executing, and percent of that sample period during which global device memory was being read or written. Neither is peak-FLOP or peak-bandwidth utilization, and neither is VRAM occupancy.

- Where the device/driver/wrapper exposes NVML timestamped sample-buffer APIs, SIA SHOULD consume new utilization/power/clock samples by source timestamp instead of increasing poll frequency; fallback polling remains available and its coarser semantics stay visible.

- Enumerate every visible physical GPU.

- Expose current clocks and relevant maximum/reference clocks where available.

- Record power usage/limit, performance state and current throttle reasons where available.

- Expose PCIe throughput/link properties where supported and record source sampling semantics.

- Use timestamped NVML running-process/process-utilization data only when supported; preserve the vendor sample timestamp and capability limitations (including configurations such as MIG where particular utilization APIs can be unsupported). Gracefully degrade to process VRAM/accounting or no GPU attribution rather than inventing zeros.

- Evaluate upgrading nvml-wrapper from the pinned 0.11.0 during dependency review, but do not upgrade solely because a newer version exists.

## 7.6 AMD provider

AMDGPU support is capability-first and SHOULD avoid requiring ROCm for basic monitoring. Stable kernel interfaces expose VRAM totals/usage and many cards expose utilization/temperature/power/clock information through DRM sysfs and hwmon. The provider MUST fix the current conceptual gap where AMD temperature sensors can be classified as GPU while GPU utilization/VRAM traces remain absent.

- Discover AMD DRM devices and map them to hwmon/PCI identity.

- Use stable amdgpu sysfs such as VRAM total/used when present.

- Use documented `gpu_busy_percent` / `mem_busy_percent` and DRM client-usage counters where present; otherwise omit utilization rather than fabricate it. Where the driver exposes a documented `gpu_metrics` snapshot, SIA MAY use it as a batched source for supported temperature/frequency/engine/power/throttle fields after semantics/version parsing is validated.

- Expose temperature, power and clocks through hwmon/sysfs where the device publishes them.

- ROCm SMI MAY be an optional enrichment provider later, but basic SIA operation must not require installing the ROCm stack.

## 7.7 Intel provider

Intel support likewise uses capability discovery. The preferred unprivileged process-level route is the kernel's standardized DRM client usage statistics in `/proc/<pid>/fdinfo/\*` where the active driver exports them: per-engine busy/cycle counters and memory-region accounting can be converted to interval utilization with their documented capacities/clock semantics. Xe has documented support; i915 and other DRM drivers may expose partially compatible keys. Device-wide frequency/throttle/temperature data come from stable sysfs/hwmon interfaces. Unsupported keys are absent, not guessed. This same DRM parser SHOULD be reusable for AMD/other DRM drivers where they implement the standardized fields.

# 8. Process and application attribution

## 8.1 Target process mode

`sia -headless` can attach to an existing PID or launch a command and profile it until exit. On kernels that support pidfds, SIA SHOULD open/retain a pidfd for the attached/launched root process so lifecycle detection is not vulnerable to PID recycling; PID plus `/proc/<pid>/stat` start time remains the recorded stable identity and fallback. Descendant discovery uses recorded PID/start-time relationships; a cgroup-based launch enclosure may be evaluated later if ordinary process-tree tracking proves insufficient, but is not required for v0.1.

```console
sia -headless --pid 12345 --output run.sia/  
sia -headless --profile --output run.sia/ -- my_application --arg value  
sia -headless --profile --output run.sia/ -- iris replay --from ...
```

## 8.2 Process metrics

- CPU utilization and CPU time for the target process; per-thread utilization is optional high-detail data.

- RSS and virtual memory; memory growth over time.

- Read/write I/O byte rates where procfs permissions allow.

- Thread count and child-process lifecycle.

- Context switches where available at acceptable cost.

- NVIDIA/other GPU process memory or utilization when the provider can attribute it.

- SIA's own process is always tracked in recorded sessions so observer cost is visible.

### 8.2.1 Failure isolation

SIA is an observer, not a supervisor. Recorder/GUI/provider failure MUST NOT kill, pause, reprioritize or reconfigure the target process. If SIA launched the target and SIA itself exits unexpectedly, the target is left running unless the user explicitly requested coupled lifecycle behavior. Process/provider permission failures degrade attribution while preserving the rest of the session.

### 8.2.2 Descendant coverage

Process-tree attribution is best-effort for descendants discovered through procfs relationships. For each session SIA records the discovery method and whether a child may have been missed/reparented before observation. The stable identity guarantee applies to processes that were observed, not to perfect discovery of every transient child. The IRIS milestone does not require cgroup creation or ptrace/eBPF solely to make descendant coverage theoretically complete.

## 8.3 Privacy/sensitive arguments

Process names and executable paths are useful; full command lines, environment variables and imported trace attributes can contain API keys, passwords, tokens or confidential filenames. Full argv/environment capture is therefore OFF by default, values known to be secret-bearing are never collected implicitly, and imported free-form attributes remain bounded. Session directories/files default to user-private permissions (0700 directory / non-world-readable files subject to platform umask). Export warns when sensitive metadata fields are present.

# 9. Application markers and trace import

## 9.1 Offline import is authoritative for IRIS

IRIS writes its own stable performance trace. For the first integration, SIA MUST import the structured JSONL span/event form plus run-manifest clock/hardware metadata; Parquet support is optional because low-volume trace events do not justify a heavy reader dependency by themselves. SIA aligns the imported artifact with a SIA session only when clock/boot identity or synchronization anchors make that defensible. This offline import is the primary IRIS integration because it creates no runtime dependency and no target-side backpressure.

## 9.2 Live marker transport

For generic applications that want live annotations, SIA SHOULD provide a local user-scoped Unix datagram endpoint under `XDG_RUNTIME_DIR` plus a `sia mark` CLI. The socket directory/file is mode-restricted to the current user and ownership is verified before binding/sending. Datagram semantics avoid target blocking; each sender SHOULD include a monotonically increasing sequence so received gaps can be noticed, and send failures are surfaced to the sender when the OS reports them. Messages/attribute maps are bounded. SIA MUST NOT claim end-to-end losslessness because the final/lone datagram can be lost without creating a sequence gap. Offline trace import remains authoritative for IRIS.

```console
sia mark begin --source iris --trace \$RUN_ID --span path-3 --category compute --name PATH_SIMULATION --attr batch=3  
sia mark instant --source iris --trace \$RUN_ID --category scheduler --name GPU_BACKEND_SELECTED --attr backend=cuda  
sia mark end --source iris --trace \$RUN_ID --span path-3
```

A language-specific SIA SDK is not required. Tiny adapters MAY be supplied for Rust/Python later. The schema is deliberately simple enough to emit from shell scripts.

## 9.3 Mapping to existing tracing standards

SIA's internal event semantics SHOULD remain convertible to Perfetto TrackEvent/Chrome-style slices, counters and instants, and Rust applications SHOULD be able to write an adapter from `tracing` spans/events. Perfetto and OpenTelemetry are not mandatory SIA dependencies in v0.1; adopting a large external tracing runtime merely to draw local stage bars would violate the simplicity objective.

# 10. Recording and session format

## 10.1 Session directory

```text
<session>.sia/  
manifest.json  
metric_descriptors.json  
entities.jsonl  
streams/  
system-0001.csv  
cpu-0001.csv  
gpu-0001.csv  
process-0001.csv  
... # schema-segmented append-only wide streams  
events.jsonl  
diagnostics.jsonl # optional derived findings  
imports/  
iris-<trace-id>.jsonl
```

The initial native storage backend is intentionally boring: buffered append-only wide CSV stream segments plus JSONL metadata/events. Each stream family carries `mono_ns`/window fields and an entity ID plus related metric columns; if its descriptor set changes, the writer closes the segment and opens the next numbered segment with a new header/schema hash. This avoids one file per process/GPU and avoids a row per metric. `sia export` produces a canonical long-form table. A binary/columnar native format remains deferred unless the Phase 0/2 observer-effect benchmarks show that text conversion/storage materially perturbs the target.

## 10.2 Manifest

```text
SessionManifest {  
schema_version  
session_id  
status + status_reasons  
sia_version + git_commit  
start/end utc  
native_clock_domain + monotonic/boottime/utc synchronization anchors  
boot_id + time_namespace_id  
host/hardware inventory fingerprint  
kernel + driver + runtime/provider versions  
SamplingPlan + resolved provider cadences/capabilities  
target process identities + discovery method/coverage notes  
writer/drop/backlog/collector-lateness summary  
live_gui_active  
imported trace ids + clock-alignment quality  
}
```

## 10.3 Crash behavior

Session files are append-only and SHOULD remain readable after an unclean exit. The manifest begins `in_progress` and ends in one explicit terminal status such as `complete`, `degraded_loss`, `recording_failed`, `environment_changed` or `aborted`; multiple flags/reasons may accompany the status. Finalization is atomic. Replay ignores only an incomplete trailing record, records `truncated_tail=true`, and never silently repairs earlier corruption. The writer uses a bounded queue and MUST NOT block collectors or the target; queue overflow/drop counts and maximum backlog are persisted. Disk-full/write failure marks recording failure, stops persistent writes safely, and never terminates the target. A session with sample/event loss beyond its predeclared completeness criterion remains viewable but is invalid for quantitative performance claims.

## 10.4 Export

- `sia export <session> --format csv` MUST export a documented long-form metric table plus event table.

- JSONL export SHOULD preserve typed event attributes and capability/status fields.

- Perfetto-compatible export MAY be added after core timeline semantics stabilize.

- Exports include schema version and never silently reinterpret units.

# 11. GUI and analysis experience

## 11.1 Preserve the overview

The existing at-a-glance overview remains a first-class view: a user should still answer 'what is hot, busy or pressured?' without configuring the profiler. The visual architecture follows a Tufte/Shneiderman-style progression: overview first; synchronized zoom/filter; details on demand. Profiling depth is revealed progressively rather than placing every available counter on the opening screen.

### 11.1.1 Information-design rules

- All time-oriented panels share one horizontal time axis and selection cursor; correlated tracks align vertically rather than using independent scrolling timebases.

- Do not mix unrelated units on one y-axis and do not use dual y-axes for quantitative comparison. Use aligned small multiples when units/scales differ.

- Aggregate first, expand on demand: package/system summaries are shown before dozens of core/thread/engine traces. If an overlay becomes hard to identify, use selection or small multiples rather than adding more colors.

- Use position and length for precise quantitative comparisons. Avoid 3-D effects, gauges/dials, decorative gradients, area/angle encodings and animation that does not represent changing data.

- Color has semantic work: reserve strong/saturated color primarily for selection, warnings and distinct resource classes; normal traces should not require a rainbow. Never make color the sole carrier of status.

- Prefer direct labels at the edge/selection and concise hover/detail text over large legends when practical. A legend is secondary navigation, not the main decoding task.

- Axes/scales may adapt, but a live scale change must be visually stable/obvious. Utilization percentages use a fixed 0-100 scale when their semantics genuinely are percentages; other scales may lock for the selected interval/session to avoid exaggerating tiny changes.

- Missing/unavailable data appear as gaps/status, never interpolated through or plotted at zero unless zero is an actual observed value.

- Keyboard navigation, scalable type/high-DPI behavior and a color-vision-deficiency-safe palette are release requirements for the new profiler views.

## 11.2 Performance timeline

The central profiler view is a synchronized timeline, not a wall of all available traces. The default profile view shows application/process lanes plus only the smallest resource set needed to orient the user (CPU demand, GPU activity/capacity context when present, memory/pressure and direct throttle/error state). Per-core, per-engine, clock, power, thermal, PCIe, disk and thread detail is revealed by selection/filtering. Zooming, panning or selecting a time range changes every visible track together. A persistent overview strip preserves context when the user zooms deeply.

### 11.2.1 Local, observer and replay source identity

The profiler header always shows the data source: LOCAL LIVE, OBSERVER LIVE <target-host/IP>, or RECORDED SESSION <session_id>. Observer mode shows target host/boot/GPU identity, pairing/authentication state, stream loss/lag and whether the session is authoritative for benchmark claims. Observer-machine CPU/GPU state is never mixed into target plots.

| Track group    | Examples                                                                                        |
|----------------|-------------------------------------------------------------------------------------------------|
| Application    | IRIS stages, imported spans, markers, benchmark phases.                                         |
| Target process | CPU, RSS, I/O, selected threads/children.                                                       |
| CPU/system     | Total/per-core utilization, frequency, PSI, load/run queue.                                     |
| GPU            | Compute utilization, memory activity, VRAM, clocks, power, temperature, throttle reasons, PCIe. |
| Memory/I/O     | RAM/swap/page faults, memory/I/O PSI, disk throughput/busy.                                     |
| SIA observer   | SIA CPU/RSS/I/O and `live_gui_active` state.                                                  |

## 11.3 Capability-aware UI

- Unsupported metrics are omitted, not plotted at zero and not left in legends.

- A device/capability panel explains why a metric is missing: unsupported, permission denied, provider unavailable or temporary read error.

- Multiple GPUs are selectable and can be displayed together or separately.

- No hard 1200x880 content minimum may make the application unusable on smaller screens; panes/scrolling/layout must respond to available space.

- Historical/live display windows operate on timestamp ranges rather than sample counts.

## 11.4 Selection details

Selecting a span or time range shows, in one compact evidence panel: what application work ran; duration/work units/throughput; target-process CPU/RSS/I/O; CPU/PSI context; GPU activity/VRAM/clocks/power/temperature/throttle evidence; transfer/I/O context; SIA observer cost; sample coverage/alignment quality; and any transparent diagnostic statements. Raw source semantics and exact values remain accessible one level deeper.

## 11.5 Interval statistics and coverage

Interval summaries MUST respect metric temporal semantics and irregular sampling. Interval-average metrics use overlap/time weighting where their observation window is known; cumulative counters use boundary/delta logic; point samples report sampled min/max/quantiles without pretending continuous coverage. SIA does not interpolate across missing/error gaps for extrema or diagnosis. Every selection detail shows sample count, covered-duration fraction where meaningful, maximum observed gap and source-resolution caveat.

For coarse vendor metrics such as NVML utilization, a short application span may contain too few independent provider samples to support a strong conclusion. SIA shows `insufficient temporal coverage` rather than presenting a precise-looking average.

## 11.6 No-manual usability gate

Before the profiler UI is accepted, engineers who did not design SIA are given representative recorded sessions and no documentation. They must be able to determine, from evidence in the UI: (1) what resource is constrained or whether none is proven; (2) which process/application span coincides with it; (3) whether GPU activity/VRAM/thermal-power state is consistent with effective use; (4) what measurement limitation weakens the conclusion; and (5) what deeper tool or code region to inspect next. All five tasks must be answerable without external instruction; completion time and wrong turns are recorded as design diagnostics, but v0.1 does not invent a universal seconds-to-answer threshold before pilot data exist.

## 11.7 Information-design review gate

Before the new profiler timeline/detail UI is merged, static wireframes or a nonfunctional prototype are reviewed jointly by the Tufte-style information-design, operator-HCI, accessibility and product-simplicity reviewers. The review explicitly asks which elements can be removed, whether every quantitative comparison has an honest scale/encoding, whether the default view contains only orientation-level information, and whether the user's eye is drawn first to the important state rather than decoration. This gate occurs before substantial polishing so visual complexity is not baked into code.

# 12. Bottleneck diagnostics

## 12.1 Philosophy

SIA diagnoses broad bottleneck classes, not kernel-level root cause. Every finding has one of three claim levels: `Observed` for a directly reported condition such as a vendor throttle reason; `Consistent with` for a multi-signal hypothesis such as CPU feeder starvation; or `Insufficient evidence`. There is no numeric confidence score. Each finding lists the exact time range, triggering observations, sampling/clock limitations and the next specialist check. The synchronized evidence remains primary; findings are annotations, not a substitute for looking at the data.

| Finding                    | Evidence pattern                                                                                                                   | Allowed wording / next step                                                                                                                                   |
|----------------------------|------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Thermal throttling         | Vendor/kernel throttle reason + temperature/clock behavior                                                                         | 'Thermal throttling observed.' Strong evidence.                                                                                                               |
| Power capping              | GPU power-limit/throttle reason + clock behavior                                                                                   | 'Power-limit throttling observed.' Strong evidence.                                                                                                           |
| CPU feeder suspicion       | Repeated GPU idle gaps during application GPU span + one/few target threads near CPU saturation                                    | 'GPU may be starved by CPU-side work.' Inspect thread/span; use perf if needed.                                                                               |
| I/O pressure               | I/O PSI stalls + target I/O + stage slowdown                                                                                       | 'I/O pressure coincides with slowdown.' Inspect storage/source.                                                                                               |
| Memory pressure            | Memory PSI/swap/major faults + stage slowdown                                                                                      | 'Memory pressure/paging coincides with slowdown.'                                                                                                             |
| Transfer-heavy GPU use     | High PCIe traffic during low/intermittent GPU compute + stage markers                                                              | 'Host/device transfer may be material.' Confirm with Nsight Systems.                                                                                          |
| Memory-heavy GPU suspicion | High vendor memory-activity metric relative to compute activity                                                                    | 'Memory activity dominates the sampled vendor signal.' This is not proof of bandwidth saturation; use Nsight Compute/vendor hardware counters for that claim. |
| VRAM capacity pressure     | Imported allocation/OOM failure, or explicit user/application memory requirement exceeding/minimally fitting sampled free capacity | 'VRAM capacity constraint observed/likely for the stated requirement.' Otherwise show minimum sampled free VRAM without an automatic pressure label.          |
| Observer effect            | SIA self resource use is material relative to target during capture                                                                | 'Capture may be perturbing workload.' Repeat headlessly/lower rate.                                                                                           |

### 12.1.1 Diagnostic trigger governance

Diagnostic thresholds must be physically defined (for example an explicit vendor throttle state), derived from hardware capacity/provider semantics, or established prospectively in the benchmark/fixture suite. They may not be tuned on a desired diagnosis. An inferred rule MUST expose its trigger logic in the UI/spec. Automatic diagnostics are not an IRIS-readiness gate; a correct trace/timeline is more valuable than a clever but unreliable label.

The diagnostic patterns below are candidate Phase 5 rules, not SIA 0.1 release requirements. Direct vendor/kernel conditions may graduate quickly; inferential patterns require fixture validation and explicit trigger definitions.

## 12.2 Deep-tool boundary

When the evidence requires scheduler traces, system calls, CPU hardware counters, CUDA kernel launch timelines or warp/cache/tensor-core metrics, SIA should recommend the appropriate deeper tool rather than duplicate it. For NVIDIA, Nsight Systems is the next step for CPU/GPU scheduling, CUDA/API and transfer timelines; Nsight Compute is the next step for kernel-level compute/memory diagnosis. Linux perf/ftrace/Perfetto are appropriate for CPU/kernel scheduling questions.

# 13. Command-line contract

| Command                                                     | Purpose                                                                                                                                                                                                                               |
|-------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sia`                                                     | Run the local live GUI/collector. Headless/observer/replay sources use the same session/GUI data model.                                                                                                                               |
| `sia -headless \[options\]`                               | Authoritative headless collector/recorder plus encrypted observer listener; no GUI/graphics initialization.                                                                                                                           |
| `sia -headless --pid <pid>`                             | Attach headless profiling to an existing process and track descendants.                                                                                                                                                               |
| `sia -headless -- <command...>`                         | Launch and profile a command headlessly until it exits while the target remains independent of SIA failure.                                                                                                                           |
| `sia view <session.sia>`                                | Open recorded session in GUI.                                                                                                                                                                                                         |
| `sia import iris <trace> --session <session>`         | Attach/import an IRIS trace to a SIA session.                                                                                                                                                                                         |
| `sia export <session> --format csv\|jsonl`              | Documented data export.                                                                                                                                                                                                               |
| `sia mark ...`                                            | Emit nonblocking local application marker/span events.                                                                                                                                                                                |
| `sia doctor`                                              | Show hardware/providers/capabilities/permissions and missing metric reasons.                                                                                                                                                          |
| `sia benchmark self`                                      | Run the registered SIA observer-effect/collection benchmark suite.                                                                                                                                                                    |
| `sia -headless --plan <name>` / `--rate <profile>`  | Select versioned SamplingPlan/profile; CLI reports actual provider cadences/resolution after capability discovery.                                                                                                                    |
| `sia -o <IP\[:port\]> <adjective-adjective-noun>`     | Run the GUI on the observer computer, authenticate with the three-word PAKE phrase printed by the target, then display the encrypted read-only live stream. `--observer` is a long-form alias.                                      |
| `sia -headless \[--listen <addr>\] \[--port <port>\]` | Run authoritative headless collector/recorder plus encrypted read-only observer listener; print target IP(s), three-word pairing phrase and exact `sia -o ...` command. `--headless` is an alias. No GUI/graphics initialization. |

## 13.1 Configuration

CLI flags override a small user config. Configuration may set sampling presets, enabled metric groups, recording path, process tracking, privacy policy and UI preferences. Hardware/provider capabilities are discovered, not configured as if they existed. The configuration format should stay human-readable.

# 14. SIA performance engineering

## 14.0 Headless implementation invariant

The `sia -headless` execution path is tested to ensure no eframe window/render loop and no wgpu/OpenGL/Vulkan device/context is initialized. GUI renderer/backend choice is benchmarked separately for local-live use, but no GUI result is used as the authoritative GPU optimization measurement unless its observer-effect equivalence gate passes.

## 14.1 SIA must profile itself

A performance profiler with opaque overhead is not trustworthy. Every recorded session includes SIA self CPU/RSS/I/O, sampling latency and queue/drop statistics. `sia benchmark self` runs fixed collector and GUI workloads and stores results in a machine-readable benchmark artifact.

## 14.2 Benchmark fixtures

| Fixture                                         | What it tests                                                                                                                                    |
|-------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| Idle machine                                    | Baseline SIA CPU/GPU/IO observer cost.                                                                                                           |
| CPU single-thread                               | Can SIA identify a saturated feeder-like thread without large perturbation?                                                                      |
| CPU multi-thread                                | Per-core/system saturation and run-queue behavior.                                                                                               |
| Memory pressure                                 | PSI, swap/fault and observer overhead under pressure.                                                                                            |
| Disk I/O                                        | Disk throughput/PSI/process I/O correlation.                                                                                                     |
| GPU compute                                     | Headless observer effect, utilization/power/clocks, process attribution.                                                                         |
| GPU transfer-heavy                              | PCIe/compute timeline and diagnostic caution.                                                                                                    |
| GPU thermal/power capped (where safe/available) | Throttle reason detection; no automatic cap modification.                                                                                        |
| Mixed CPU/GPU pipeline                          | Stage/marker alignment and starvation diagnosis.                                                                                                 |
| Long recording                                  | Memory growth, writer stability, file size, sample loss and crash recovery.                                                                      |
| Repeated before/after comparison                | Randomized/interleaved control-vs-SIA trials, effect-size uncertainty and thermal/background-state handling.                                     |
| Suspend/resume                                  | Clock-domain anchors detect suspend; ordinary benchmark claim is marked invalid/environment_changed.                                             |
| Short-burst/aliasing                            | Demonstrate that coarse sampled metrics may miss sub-window bursts and that SIA reports temporal-coverage limitations rather than false absence. |

## 14.3 Regression gate

Performance-sensitive changes are compared with the most recent accepted benchmark on the same hardware/profile using the predeclared repeated-trial design. A statistically and practically material regression in observer cost, sampling jitter/loss, GUI responsiveness or recording throughput blocks merge unless an ADR explains the trade. A tiny difference whose uncertainty spans the equivalence region is not called a regression or speedup. Benchmark criteria are never tuned against IRIS financial outcomes.

# 15. Security and permissions

- Default operation runs as the logged-in user; SIA MUST NOT require the whole application to run as root.

- Unavailable privileged metrics are omitted with a clear permission explanation.

- The live marker socket is user-scoped under XDG_RUNTIME_DIR and created with user-only permissions.

- Recorded session files default to user-only permissions because process paths/markers may be sensitive.

- Environment variables are not collected by default; full argv is opt-in.

- A later remote web viewer binds to loopback by default. Remote binding requires explicit configuration and an authentication design; this is not part of IRIS readiness.

- SIA does not change GPU clocks, power limits, CPU governors or target-process affinity in v0.1.

## 15.1 Untrusted local inputs

Marker datagrams, imported traces, session files and vendor/sysfs text are treated as untrusted input. Parsers use bounded record/attribute sizes, checked integer conversion and no panics on malformed fields. Relative paths in imports cannot escape the session directory. A malformed provider/import disables only that stream/record with an error status; it must not crash the target application.

## 15.2 Pairing, PAKE and observer-stream security

SIA 0.1 observer live viewing exposes only a read-only performance stream and no remote-control API. The user-visible three-word phrase is a session-scoped PAKE password, not a bulk-encryption key or truncated public-key fingerprint. A successful standard PAKE plus key confirmation yields strong ephemeral session keys; all subsequent telemetry is authenticated and encrypted. Passive capture must not permit offline phrase testing. The phrase-space/attempt policy is derived from the registered local-LAN threat model rather than a fixed entropy number. Wrong phrases, modified handshakes or modified ciphertext fail closed.

Stream frames are bounded, versioned, sequence-numbered and parsed as untrusted input. Length limits, schema/version negotiation, rate limits, replay/tamper rejection and malformed-frame handling are mandatory. The phrase is never persisted in session files or ordinary logs. Process command lines and other sensitive metadata follow the configured privacy policy. There is no plaintext observer fallback in the released 0.1 product.

# 16. Verification and acceptance tests

Verification ownership is explicit. A test belongs to the lowest class that can decide it reliably: deterministic code tests first, AI-agent system tests where orchestration/inspection is required, and human tests only for genuinely subjective/usability/support judgments. An AI agent may run code tests, but that does not convert a deterministic test into an AI-judged test. Hardware-unavailable providers cannot be certified by simulation.

## 16.1 Code-automated tests

| Test                                   | Acceptance criterion                                                                                                                                                                                                                                                                                    |
|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Timestamp truth                        | Synthetic irregular collection delays preserve actual monotonic observation timestamps/windows; no nominal-period drift.                                                                                                                                                                                |
| Sampling/GUI/remote decoupling         | Artificial GUI and remote-stream stalls do not alter collector deadlines/timestamps; secondary consumers may lose display frames without blocking collection.                                                                                                                                           |
| No idle 60-Hz redraw requirement       | With no interaction/new sample, GUI repaint/CPU activity falls to the sample/event-driven baseline rather than the current continuous 16-ms behavior.                                                                                                                                                   |
| Plot-cache regression                  | Repeated redraws without new samples do not repeatedly allocate/copy full visible history beyond the registered GUI budget.                                                                                                                                                                             |
| Unsupported metric behavior            | Missing provider capability produces absent/unavailable data, never a fabricated zero trace.                                                                                                                                                                                                            |
| NVIDIA metric schema semantics         | Mock/fixture distinguishes GPU busy, memory-subsystem activity and VRAM occupancy fields and comparability groups.                                                                                                                                                                                      |
| Multi-GPU identity                     | Synthetic two-device provider preserves UUID/PCI identity and never aliases devices by ordinal alone.                                                                                                                                                                                                   |
| AMD provider contract fixture          | AMDGPU sysfs/hwmon fixtures parse documented fields and degrade cleanly when absent; this does not constitute hardware verification.                                                                                                                                                                    |
| Intel-GPU provider contract fixture    | Intel DRM/sysfs fixtures parse documented fields and degrade cleanly when absent; this does not constitute hardware verification.                                                                                                                                                                       |
| Process PID reuse                      | Synthetic procfs fixture reuses PID with a new start time and creates a new ProcessIdentity.                                                                                                                                                                                                            |
| Child tracking                         | Integration fixture tracks launched descendants without confusing unrelated reused PIDs.                                                                                                                                                                                                                |
| Marker nonblocking/loss                | Flood marker receiver/sender; target side never blocks and sequence/drop accounting is explicit.                                                                                                                                                                                                        |
| IRIS trace alignment                   | Synthetic known-clock IRIS spans align to known metric impulses; wall-only/custom-clock fallback exposes reduced alignment quality.                                                                                                                                                                     |
| Session crash recovery                 | Forced termination yields readable valid prefix, unclean terminal state and incomplete-span status.                                                                                                                                                                                                     |
| Dropped-sample/backpressure accounting | Forced writer/stream backlog increments drop/backlog counters and degrades data-quality status without blocking collector/target.                                                                                                                                                                       |
| PSI parsing/aggregation                | Synthetic CPU/memory/I/O PSI preserves `some`/`full`, total-delta and gap/error semantics.                                                                                                                                                                                                          |
| Clock-domain alignment                 | Matching boot/CLOCK_MONOTONIC anchors align exactly; ambiguous domains are approximate/rejected for fine correlation.                                                                                                                                                                                   |
| Temporal semantics preservation        | Interval average/delta/vendor-sampled records preserve observation windows rather than being mislabeled instantaneous.                                                                                                                                                                                  |
| PIDFD lifecycle                        | Where pidfd is available, lifecycle follows the original process; PID/start-time fallback remains correct.                                                                                                                                                                                              |
| Cross-vendor metric semantics          | Provider percentages with different definitions remain distinct descriptors/comparability groups.                                                                                                                                                                                                       |
| Absolute-deadline scheduler            | Injected collector delays record lateness, skip missed ticks and do not cause catch-up bursts or accumulated drift.                                                                                                                                                                                     |
| Interval aggregation                   | Irregular/windowed samples produce correct overlap/time-weighted summaries and coverage.                                                                                                                                                                                                                |
| Schema-segment recording               | Metric/device additions/removals create new schema segments without reinterpreting old records.                                                                                                                                                                                                         |
| True headless path                     | `sia -headless` initializes no egui/eframe/wgpu/Vulkan/OpenGL rendering path; observer GUI runs in a separate process/machine.                                                                                                                                                                        |
| Provider loss/reset isolation          | Injected provider loss/sysfs disappearance creates gaps/lifecycle events while unrelated collectors continue.                                                                                                                                                                                           |
| Malformed input containment            | Malformed/oversized remote frames, markers, session rows and provider strings fail boundedly without panic/path traversal.                                                                                                                                                                              |
| Session terminal states                | Clean, dropped-sample, disk-full, environment-change and abort paths persist explicit terminal states/reasons.                                                                                                                                                                                          |
| IRIS JSONL minimum adapter             | Representative IRIS v21 JSONL trace/manifest imports without requiring Parquet.                                                                                                                                                                                                                         |
| Observer protocol/version negotiation  | Compatible framed-stream schema succeeds; incompatible protocol/schema version fails explicitly before data are misinterpreted.                                                                                                                                                                         |
| Observer backpressure isolation        | Artificially stall the observer socket/GUI; target collector and local recorder continue on schedule and observer loss/lag is reported.                                                                                                                                                                 |
| Observer timestamp integrity           | Observer GUI preserves target monotonic/boot timestamps and never substitutes observer clock.                                                                                                                                                                                                           |
| Observer frame/parser containment      | Malformed/oversized handshake or encrypted stream frames are rejected boundedly without panic, allocation explosion or mutation of target state.                                                                                                                                                        |
| Three-word PAKE pairing                | Correct phrase derives matching session keys and succeeds; wrong phrase fails closed. Phrase is never sent in cleartext or used directly as a bulk key. Standard protocol test vectors pass.                                                                                                            |
| Pairing phrase entropy/grammar         | CSPRNG selection is uniform over the versioned adjective/adjective/noun lists; code computes phrase-space size and verifies it exceeds the security-ADR requirement derived from online-attempt policy/threat target; canonicalization is unambiguous and banned/confusable entries cannot be selected. |
| PAKE transcript secrecy                | Published PAKE test vectors pass; test harness confirms no phrase/plaintext telemetry appears in handshake bytes and captured transcript cannot be validated by a simple offline candidate-check API.                                                                                                   |
| Key confirmation                       | Both peers must confirm the PAKE-derived key before telemetry is accepted; transcript/role mismatch and wrong phrase fail.                                                                                                                                                                              |
| AEAD tamper/replay                     | Bit changes, frame replay, sequence rollback and wrong-session keys are rejected; valid encrypted frames round-trip.                                                                                                                                                                                    |
| Pairing attempt limit                  | Failed online pairings trigger the declared bounded-attempt/backoff policy without blocking collection/recording.                                                                                                                                                                                       |
| Secret non-persistence                 | Pairing phrase/derived keys do not appear in session manifest, CSV/JSONL exports, normal logs or crash metadata.                                                                                                                                                                                        |
| Exact command rendering                | Headless-generated observer command round-trips through CLI parser for IPv4/IPv6/port-fallback forms and the printed phrase.                                                                                                                                                                            |
| Observer single-client/reconnect       | Only one active observer is accepted in 0.1; second concurrent connection is rejected without stream/collector disruption; after disconnect the same session phrase reconnects until headless session end.                                                                                              |
| Observer CLI secret handling           | Observer parses the phrase, minimizes its lifetime, and redacts/overwrites its own process argument representation where the OS/runtime permits; tests document any platform limitation. No phrase enters SIA logs.                                                                                     |
| Headless output contract               | Given single/multiple interface and default/fallback-port fixtures, startup prints host, recording path, phrase and at least one complete parseable `sia -o` command; no hidden operator inference is required.                                                                                       |

## 16.2 AI-agent executable system tests

These tests require launching workloads, operating the OS/tools or visually/logically inspecting resulting sessions, but do not require subjective human judgment. They can be executed by an AI coding/operations agent when it has shell/GUI access to the required machine.

| Test                                 | Acceptance criterion                                                                                                                                                                                                             |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Observer-effect equivalence          | Run repeated/interleaved control versus headless-capture trials on the Intel-CPU/NVIDIA-GPU machine and analyze whether predefined equivalence margins pass.                                                                     |
| GPU GUI observer test                | Compare NVIDIA GPU workload under headless capture versus live local GUI; quantify perturbation and label GUI authoritative only if margin passes.                                                                               |
| Encrypted-observer effect            | Compare target-local headless capture with and without a successfully paired encrypted observer on representative CPU/GPU workloads; quantify PAKE setup and steady-state encryption/streaming observer cost.                    |
| NVIDIA provider cross-check          | On real NVIDIA hardware, cross-check identity, VRAM, utilization, temperature, clocks and power against `nvidia-smi`/NVML semantics within timestamp tolerance.                                                                |
| Provider sample-buffer fidelity      | On supported NVIDIA hardware, verify timestamped native samples are consumed once and fallback polling is labeled/coarser.                                                                                                       |
| GPU starvation diagnosis fixture     | Run controlled CPU-feeder/GPU-workloads; confirm captured evidence shows the expected CPU saturation/GPU idle pattern without overstated diagnosis.                                                                              |
| Suspend/resume hardware check        | Perform or orchestrate one actual suspend/resume on the primary Linux machine and verify environment_changed/clock discontinuity behavior.                                                                                       |
| Disk-full/crash hardware run         | Exercise real temporary-filesystem ENOSPC and process-kill cases and confirm target process survives and session prefix replays.                                                                                                 |
| Responsive GUI agent inspection      | Run representative local/remote/replay sessions at several window sizes; inspect screenshots/interactions for clipping, dead traces and source-identity labeling.                                                                |
| Visualization integrity agent review | Inspect profiler screenshots/session interactions for synchronized x-axes, visible gaps/coverage, no mixed-unit dual axes and stable rescaling cues.                                                                             |
| Observer two-host flow               | Start `sia -headless` on a target, use the exact printed `sia -o <IP> <phrase>` command from a second host, interrupt/reconnect, and verify explicit encrypted stream/loss/terminal behavior with no graphics on target. |
| IRIS observer workflow               | Run or replay an IRIS-like staged workload on the target and verify the observer GUI can select a stage and inspect aligned CPU/GPU/memory/pressure evidence.                                                                    |
| Performance regression benchmark     | Run the frozen SIA benchmark suite before/after performance-sensitive changes and flag statistically material overhead regressions.                                                                                              |
| Encrypted-observer packet inspection | Capture observer traffic with tcpdump/Wireshark during a two-host run; verify phrase and telemetry values/process names are not visible in plaintext, tampering/connection loss are handled explicitly.                          |

## 16.3 Human-required tests

These gates deliberately require human judgment or real-world support ownership. They must not be silently 'passed' by an AI agent because a screenshot looked plausible.

| Test                            | Acceptance criterion                                                                                                                                                                                                                                                      |
|---------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tufte/information-design review | A human information-design reviewer approves the default overview/timeline hierarchy, scales, labels, progressive disclosure and absence of chartjunk before final profiler UI implementation.                                                                            |
| No-manual diagnosis             | Experienced engineers who did not design SIA answer the five specified diagnosis/navigation questions from representative sessions without documentation; failures are UI defects.                                                                                        |
| Observer-use usability          | A human on one computer types/copies the exact `sia -o <IP\[:port\]> <phrase>` command printed by a second Linux/NVIDIA target and can identify target identity, benchmark-authority status, saturation/waiting and responsible application stage without a manual. |
| Accessibility/keyboard review   | A human verifies keyboard-only navigation, focus/selection clarity, readable scaling and color-not-sole encoding; automated checks are supporting evidence only.                                                                                                          |
| Hardware-support certification  | A support claim for a GPU vendor/model family requires at least one real device test by a human or supervised agent plus preserved session/capability evidence. Fixture-only backends remain hardware-unverified.                                                         |
| Release sign-off                | A human reviewer inspects the benchmark report, known limitations, unsupported/unverified hardware labels and representative GUI sessions before declaring SIA 0.1 supported.                                                                                             |
| Pairing usability               | Human reviewer starts headless SIA, reads one complete displayed command, types/copies it on a second computer and connects without documentation. Review also rejects obscure, confusable, offensive or error-prone phrase vocabulary.                                   |

## 16.4 Hardware coverage and support status

The development/test inventory available for SIA 0.1 is an Intel CPU plus NVIDIA GPU. Therefore the initial hardware certification target is Linux on Intel/x86-64 CPU with NVIDIA GPU. AMD GPU and Intel GPU providers may be implemented in the same capability architecture, compiled, unit-tested and exercised against synthetic/sysfs fixtures, but those activities cannot establish hardware support. Until a real device is made available, the UI/docs MUST label those backends Implemented - hardware unverified (or Experimental) and the IRIS-ready release gate MUST NOT depend on them.

| Platform/provider                                    | 0.1 implementation                                            | Required verification                                                      | Release status                                       |
|------------------------------------------------------|---------------------------------------------------------------|----------------------------------------------------------------------------|------------------------------------------------------|
| Intel CPU + NVIDIA GPU (available)                   | Full CPU/Linux + NVIDIA provider                              | Code tests + AI-agent hardware suite + human release sign-off              | Primary supported/certified platform                 |
| CPU-only path on available Linux host                | Full CPU/memory/pressure/process path; NVIDIA may be disabled | Code/AI suite with GPU provider disabled                                   | Supported where tested                               |
| AMD GPU (not available)                              | Capability-driven provider may be implemented                 | Compile/unit/sysfs fixtures only until real AMD hardware is supplied       | Implemented - hardware unverified; not a 0.1 blocker |
| Intel GPU (not available unless separately supplied) | Capability-driven provider may be implemented                 | Compile/unit/sysfs fixtures only until real Intel GPU hardware is supplied | Implemented - hardware unverified; not a 0.1 blocker |
| Other Linux hardware                                 | Graceful capability discovery/absence                         | No support claim without real-device evidence                              | Unsupported/unverified until tested                  |

# 17. Implementation plan and gates

The phases are intentionally ordered so the profiler exists before it is trusted to optimize IRIS. No GitHub PR should implement a later phase while an earlier gate that changes measurement semantics remains unresolved.

| Phase                                                | Scope                                                                                                                                                                                                                                                                                                                   | Exit gate                                                                                                                                                                                                                                                                                                                                       |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Phase 0 - baseline and test harness                  | Benchmark current SIA on the available Intel-CPU/NVIDIA-GPU hardware; freeze ObserverEffectTest margins and repeated-trial methodology; create synthetic procfs/sysfs/NVML/DRM/provider fixtures; define remote-stream/clock/suspend fixtures; establish usability baseline and test ownership classes.                 | Approved benchmark ADR; current 16-ms repaint/1-Hz collector overhead quantified; code/AI/human test matrices frozen; observer-effect and timestamp/alignment criteria frozen prospectively.                                                                                                                                                    |
| Phase 1 - low-overhead architecture                  | Separate collector/model/GUI modules; actual clock-domain timestamps; absolute-deadline scheduler; source-native counter/sample-buffer abstraction; sample-driven repaint; plot caching/decimation; capability model; no-graphics headless execution path.                                                              | Issue #1/#3/#4 class problems materially improved; headless-independent collector tests pass; `sia -headless` initializes no graphics device; scheduler/provider failure-isolation fixtures pass.                                                                                                                                            |
| Phase 2 - session recording and headless mode        | `sia -headless`; append-only local recording; encrypted read-only observer transport; three-word CSPRNG phrase + standard PAKE/key confirmation; offline replay; crash/disk-full behavior; self-overhead metrics.                                                                                                     | Target prints usable IP(s), phrase and exact `sia -o` command; observer connects with no graphics on target; wrong phrase/tamper fail closed; stream stalls do not perturb collector/recorder; headless and encrypted-observer effects are measured.                                                                                          |
| Phase 3 - process attribution and application events | PID/start-time identity + pidfd lifecycle where available, child tracking, proc I/O/RSS/CPU; Unix marker receiver/CLI; clock-domain-aware generic spans; IRIS trace importer with alignment quality.                                                                                                                    | Known marker/process/clock fixtures pass; no target blocking; IRIS trace aligns exactly only when clock contract is satisfied.                                                                                                                                                                                                                  |
| Phase 4 - NVIDIA profiler + IRIS-ready UI            | Enumerated NVML provider; exact utilization/memory semantics; native sample buffers where supported; VRAM, power, clocks, temp, throttle, PCIe, process attribution; IRIS JSONL importer; responsive evidence-first local/remote/replay timeline. Tufte/HCI wireframe review precedes final profiler-UI implementation. | IRIS-ready on available Intel/NVIDIA platform: authoritative target-local `-headless` recording + replay and live paired/encrypted `-o` GUI correlate IRIS stages with CPU/GPU/VRAM/power/throttle/pressure evidence; code + AI hardware + human pairing/visual/no-manual gates pass. AMD/Intel-GPU hardware certification is not required. |
| Phase 5 - transparent diagnostics                    | Only after IRIS-ready capture/UI: add evidence-backed `Observed` / `Consistent with` findings, exact trigger exposition and deep-tool recommendations. Keep raw synchronized evidence primary and diagnostics fully disableable.                                                                                    | Diagnostic rules recover known fixture conditions without overstating sampled evidence; no opaque score; disabling diagnostics leaves full profiler functionality.                                                                                                                                                                              |
| Phase 6 - AMD/Intel capability providers             | AMDGPU and Intel DRM/sysfs providers may be implemented behind the common capability interface; fixtures/documented API semantics are mandatory.                                                                                                                                                                        | Backends compile and pass provider fixtures, but remain explicitly hardware-unverified until real devices are supplied. No fake support claim and no SIA 0.1/IRIS dependency on unavailable hardware.                                                                                                                                           |
| Phase 7 - optional browser/web viewer                | Only if still desired after native observer-GUI experience: explicitly started secure read-only browser/web UI.                                                                                                                                                                                                         | Separate security/performance review; not required because native SIA observer GUI is core.                                                                                                                                                                                                                                                     |

## 17.1 Suggested PR decomposition after specification approval

The following is a future decomposition, not authorization to create the PRs now:

1\. Baseline benchmark harness and observer-effect ADR.

2\. Collector/clock/data-model extraction with no user-visible feature expansion.

3\. Sample-driven GUI repaint + plot cache/decimation + responsive layout.

4\. Capability model and multi-GPU identity; fix unsupported/dead traces.

5\. Headless record/view/session format + CSV export.

6\. Process tree attribution.

7\. Application marker schema, Unix receiver and `sia mark`.

8\. IRIS trace importer.

9\. Expanded NVIDIA NVML provider.

10\. Profiler timeline/details and first evidence-based diagnostics.

11\. AMDGPU provider.

12\. Intel provider.

13\. Optional remote web viewer, only after a separate review.

# 18. IRIS integration contract

| IRIS v21 requirement                             | SIA behavior                                                                                                                                                                                                                                                           |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| RunId + stage/workload spans                     | Import mandatory JSONL span/events and display as application lanes; preserve attributes. Parquet adapter is optional.                                                                                                                                                 |
| Monotonic start/end timestamps                   | Align by boot/clock identity; show alignment quality.                                                                                                                                                                                                                  |
| CPU/GPU/backend identity                         | Display alongside system/provider metrics for the interval.                                                                                                                                                                                                            |
| Batch/work-unit count                            | Show in selection details; can normalize throughput by work units.                                                                                                                                                                                                     |
| Hardware/runtime/profile id                      | Store original trace metadata and session hardware fingerprint.                                                                                                                                                                                                        |
| Numerical/result status                          | Show as span attribute; SIA does not interpret financial correctness.                                                                                                                                                                                                  |
| Device memory / transfer metadata when available | Overlay/import beside SIA's independently sampled GPU/PCIe observations.                                                                                                                                                                                               |
| No SIA API dependency                            | Offline import is sufficient; live marker transport optional.                                                                                                                                                                                                          |
| Clock domain / boot identity                     | Required for exact offline alignment. Matching Linux CLOCK_MONOTONIC + boot/time namespace aligns directly; custom clocks require sync anchors; ambiguous traces show reduced alignment quality.                                                                       |
| Off-target observer development/inspection       | IRIS machine runs `sia -headless`; a second workstation runs the exact printed `sia -o <IP> <phrase>` command. Observer stream is PAKE-authenticated/encrypted; authoritative backend decisions use target-local recording unless observer equivalence passes. |

## 18.1 Throughput comparisons

SIA MAY compute implementation-level throughput such as work_units/second for an application span and compare repeated sessions on the same workload/hardware identity. It MUST NOT declare an IRIS configuration financially superior. Any comparison that changes numerical precision or result status is flagged as non-equivalent rather than treated as a speedup.

# 19. Decisions deliberately deferred until Phase 0/implementation evidence

| Decision                                   | Current specification position                                                                                  | Why deferred                                                                                                          |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| Default monitor/profile sampling intervals | Benchmark-derived, then frozen in config defaults                                                               | Source resolutions and observer cost vary by machine/provider.                                                        |
| Native storage beyond CSV/JSONL            | Do not add yet                                                                                                  | Only change if measured recording volume/latency justifies it.                                                        |
| nvml-wrapper upgrade                       | Evaluate during dependency review; upgrade if required for timestamped sample/process/device APIs actually used | Current pin is not sacred, but dependency churn must buy a specified capability or bug fix.                           |
| Per-thread sampling default                | Optional                                                                                                        | Can add cost on high-thread-count applications.                                                                       |
| Perfetto export                            | Optional future adapter                                                                                         | Useful interoperability, but not needed for IRIS readiness.                                                           |
| Browser/web remote viewer                  | Deferred; native `sia -o <IP>` GUI is core for 0.1                                                          | Avoid a second UI/protocol stack until native observer mode proves insufficient.                                      |
| Privileged Intel/CPU hardware counters     | Not v0.1 core                                                                                                   | Use perf/VTune/deeper tools; avoid root/helper complexity.                                                            |
| Higher-rate GPU engine metrics             | Provider/tool dependent                                                                                         | Do not pretend NVML coarse utilization is Nsight-quality kernel profiling.                                            |
| Cgroup-enclosed target launch              | Deferred; pidfd root + PID/start-time tree first                                                                | Useful for robust descendants but adds systemd/cgroup complexity; adopt only if measured tree tracking is inadequate. |

# 20. Review committee

| Reviewer                                | Required challenge                                                                                                                                                                                                                                                                                            |
|-----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Occam/system architect                  | Does each subsystem solve a demonstrated profiling need? Prefer one executable, modules, files and capability discovery over services/plugins/databases.                                                                                                                                                      |
| Measurement-science reviewer            | Can observer effect, timestamp truth, sampling jitter and data loss be measured and bounded? Are claims stronger than the counters justify?                                                                                                                                                                   |
| Linux performance reviewer              | Are PSI, procfs, sysfs and process identities used correctly? Does SIA stop before reimplementing perf/ftrace?                                                                                                                                                                                                |
| Rust/egui reviewer                      | Can collection run independently of UI? Are needless repaints/allocations removed? Is the architecture testable without a GUI?                                                                                                                                                                                |
| NVIDIA reviewer                         | Are NVML utilization, memory activity, VRAM, PCIe, power, clocks and throttle semantics kept distinct? Is Nsight boundary clean?                                                                                                                                                                              |
| AMD/Intel reviewer                      | Does capability-driven design avoid NVIDIA assumptions and gracefully handle partial stable APIs?                                                                                                                                                                                                             |
| Observability/tracing reviewer          | Are spans/events/timestamps portable and sufficiently standard to import IRIS/Rust tracing/Perfetto later without a heavy runtime dependency?                                                                                                                                                                 |
| UX reviewer                             | Does SIA remain immediately understandable and responsive rather than becoming another noisy monitoring dashboard?                                                                                                                                                                                            |
| IRIS consumer reviewer                  | Can SIA align IRIS stages with machine state and measure CPU/GPU backend changes without becoming an IRIS dependency?                                                                                                                                                                                         |
| Security/privacy reviewer               | Does the three-word UX use a standard PAKE rather than low-entropy keying/fingerprint truncation? Is bulk telemetry strongly authenticated/encrypted, phrase never persisted, offline guessing resisted, online guessing bounded, parser/network surface small, and Internet exposure explicitly unsupported? |
| Tufte-style information-design reviewer | Does every plotted mark earn its place? Are time axes aligned, units honest, quantitative comparisons encoded with position/length, overlays limited, color purposeful, legends/minutiae reduced, and details progressively disclosed?                                                                        |
| Operator HCI / no-manual reviewer       | Can an experienced engineer unfamiliar with SIA answer what is saturated, what is waiting, which process/stage caused it, whether the GPU is being used effectively, and what to inspect next without reading documentation?                                                                                  |
| Benchmarking/statistics reviewer        | Are observer-effect and before/after claims based on repeated interleaved experiments, effect sizes and uncertainty rather than single runs or backtest-like tuning of thresholds?                                                                                                                            |
| Reliability/operations reviewer         | What happens on disk full, writer backlog, process exit/PID reuse, GPU reset/hotplug, suspend/resume, SIA crash and malformed/truncated sessions? Can the profiler ever harm the target?                                                                                                                      |
| Cross-vendor semantic reviewer          | Are similar-looking percentages kept distinct when NVIDIA/AMD/DRM definitions differ? Is commonality capability-based rather than lowest-common-denominator fiction?                                                                                                                                          |
| Product/simplicity custodian            | Protect SIA's original clean-monitor identity. Which features should be removed, deferred or exposed only on demand? Is SIA becoming an inferior Nsight/perf clone?                                                                                                                                           |
| Accessibility reviewer                  | Can the profiler be used with keyboard navigation, high-DPI/font scaling and common color-vision deficiencies? Is meaning ever encoded by color alone?                                                                                                                                                        |

## 20.0 Review weighting

Measurement science, Occam/system simplicity, Linux performance, GPU semantics and Tufte-style information design carry the greatest weight because errors there can make the profiler misleading or self-defeating. Security/reliability/cross-vendor reviewers have veto authority on unsafe or falsely portable claims. Domain specialists may request more metrics, but the Tufte/product/no-manual reviewers may remove or hide them when they do not improve diagnosis.

## 20.1 Committee consensus

Hostile pass 2 resolves the remaining implementation/security ambiguity without changing the user experience. Measurement science accepts encrypted observer streaming because authoritative recording remains local and the added path is benchmarked. Security accepts the three-word phrase only through a standard PAKE that prevents offline guessing and derives strong ephemeral AEAD keys; exact library choice is deferred to a security ADR rather than pretending an unaudited crate is approved. Occam/product reviewers delete live phrase-regeneration/account/certificate machinery and retain one observer, reconnect-with-same-session-phrase, and restart-on-compromise. Linux/network reviewers accept direct LAN observation with conservative interface discovery and explicit bind overrides. Tufte/HCI reviewers accept the normative copy/paste output and optional prompt mode. The committee is converged on the architecture and verification plan.

# 21. SIA 0.1 IRIS-ready acceptance criteria

- True headless recording (no GUI/graphics device), offline replay, explicit session terminal state and transparent CSV/JSONL export work on the primary Linux/NVIDIA machine.

- Observer live sessions visibly identify target host/boot/device, pairing/encryption state, stream loss/lag and benchmark-authority status. Performance claims use target-local headless recording unless encrypted-observer equivalence has been demonstrated.

- A SIA GUI on another computer can connect using the exact target-printed `sia -o <IP> <three-word-phrase>` command; the observer authenticates through the declared PAKE and receives authenticated encrypted telemetry while the target initializes no graphics stack.

- Collector timestamps use the declared Linux clock domain, actual observation windows/lateness and remain independent of GUI frame rate; suspend/ambiguous imported clocks are detected and labeled.

- Headless observer-effect equivalence margins frozen in Phase 0 pass on the CPU/GPU benchmark suite.

- Live GUI no longer requires continuous 16-ms repaint when no new data/user interaction exists.

- Unsupported hardware metrics disappear cleanly; no dead GPU/VRAM legend entries.

- NVIDIA provider enumerates all visible devices, separates kernel-busy utilization, global-memory activity and VRAM occupancy, and consumes timestamped native samples where supported without pretending they equal peak compute/bandwidth percentages.

- NVIDIA temperature, clocks, power/throttle and PCIe metrics are recorded where supported; unsupported calls are capability results.

- PSI CPU/memory/I/O, target-process CPU/RSS/I/O and SIA self-overhead are available where the kernel permits.

- `sia -headless -- <command>` and `--pid` retain race-resistant identity for the observed root/descendant processes (pidfd where available, PID+start-time fallback); limitations in descendant discovery are reported rather than hidden.

- Mandatory IRIS v21 JSONL trace import aligns RunId/stage spans exactly only when clock-domain/boot synchronization is valid and otherwise displays reduced alignment quality; Parquet support is not an IRIS-ready prerequisite.

- Application marker emission is nonblocking; sender sequence gaps/send failures are recorded when observable, and live markers are never treated as lossless authoritative instrumentation.

- Profiler timeline uses synchronized shared time axes, progressive disclosure and interval statistics with visible coverage/clock quality; selecting an IRIS/application span exposes CPU/GPU/memory/I/O/throttle evidence without mixed-unit/dual-axis distortion.

- The IRIS-ready gate does not depend on automatic bottleneck diagnoses; disabling diagnostics still leaves all authoritative capture, replay and evidence-inspection capability.

- Session crash/disk-full/provider-reset recovery, explicit degraded terminal states, bounded input handling and dropped-sample/backlog accounting pass.

- Responsive/small-screen, keyboard/accessibility, color-not-sole-encoding and no-manual usability gates pass for the profiler views.

- No requirement for IRIS code to link a SIA library or keep SIA running in production.

- The release report separates code-automated, AI-agent-executable and human-required gates; none of the human information-design/no-manual/support gates may be self-certified by an AI agent.

- Intel-CPU/NVIDIA-GPU is the primary hardware-certified platform for 0.1. AMD/Intel-GPU provider code, if present without real-device testing, is explicitly labeled hardware-unverified/experimental and does not block IRIS readiness.

- The headless terminal prints observed-machine identity, local recording path, three-word phrase and at least one complete `sia -o <IP\[:port\]> <phrase>` command that succeeds without documentation.

- One active observer and disconnect/reconnect behavior are deterministic; observer failures or backoff never interrupt target-local recording.

# 22. Post-0.1 roadmap

- AMD and Intel provider completeness based on real donated/test hardware.

- Read-only remote live web viewer/headless server if the original README objective remains useful after local recording exists.

- Session comparison UI: align two equivalent benchmark sessions and show interval-by-interval deltas with numerical-result equivalence status.

- Perfetto export/import adapter and perhaps launch helpers for Nsight/perf captures.

- Optional deeper CPU process counters through perf_event_open only if demand justifies permission/overhead complexity.

- Long-duration storage compression/downsampling if real recordings prove CSV/JSONL insufficient.

- Additional accelerator providers (Intel NPU/other AI accelerators) through the same capability model.

# 23. References and source basis

- SIA repository baseline and README/issues: https://github.com/davecrawley/sia — main reviewed at commit 524ecd52746419131b5209364f327c09d0ced114.

- IRIS — Investment Research and Intelligence System Specification v21, Section 26 Performance and observability contract (internal working document, 26 August 2026).

- NVIDIA, NVML API Reference Guide: https://docs.nvidia.com/deploy/nvml-api/

- NVIDIA, NVML device query/sample-buffer APIs (`nvmlDeviceGetSamples`, utilization/process/device queries): https://docs.nvidia.com/deploy/nvml-api/group\_\_nvmlDeviceQueries.html

- NVIDIA, Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/

- Linux kernel, Pressure Stall Information: https://docs.kernel.org/accounting/psi.html

- Linux kernel, DRM client usage statistics: https://docs.kernel.org/gpu/drm-usage-stats.html

- Linux kernel, AMDGPU VRAM/sysfs information: https://docs.kernel.org/gpu/amdgpu/driver-misc.html

- Linux kernel, AMDGPU power/thermal monitoring (`gpu_busy_percent`, `mem_busy_percent`, `gpu_metrics`): https://docs.kernel.org/gpu/amdgpu/thermal.html

- Linux kernel, Intel Xe DRM client usage/frequency documentation: https://docs.kernel.org/gpu/xe/xe-drm-usage-stats.html and https://docs.kernel.org/gpu/xe/xe_gt_freq.html

- Linux man-pages, `pidfd_open(2)`: https://man7.org/linux/man-pages/man2/pidfd_open.2.html

- Linux clock APIs / suspend semantics (`CLOCK_MONOTONIC`, `CLOCK_BOOTTIME`): https://man7.org/linux/man-pages/man2/timerfd_create.2.html

- Perfetto, Track Events: https://perfetto.dev/docs/instrumentation/track-events

- Perfetto, synchronization of multiple clock domains: https://perfetto.dev/docs/concepts/clock-sync

- Rust `tracing` crate: https://docs.rs/tracing/latest/tracing/

- Tomas Kalibera & Richard E. Jones, “Rigorous Benchmarking in Reasonable Time,” ISMM 2013, DOI 10.1145/2464157.2464160.

- William S. Cleveland & Robert McGill, “Graphical Perception: Theory, Experimentation, and Application to the Development of Graphical Methods,” JASA 79 (1984), DOI 10.1080/01621459.1984.10478080.

- Ben Shneiderman, “The Eyes Have It: A Task by Data Type Taxonomy for Information Visualizations,” IEEE Visual Languages 1996.

- Edward R. Tufte, The Visual Display of Quantitative Information, 2nd ed., Graphics Press, 2001.

- egui `Context::request_repaint_after` documentation: https://docs.rs/egui/latest/egui/struct.Context.html

RFC 9382, “SPAKE2, a Password-Authenticated Key Exchange,” September 2023: https://www.rfc-editor.org/rfc/rfc9382.html

RFC 9807, “The OPAQUE Augmented Password-Authenticated Key Exchange (aPAKE) Protocol,” July 2025 (alternative/reference PAKE design): https://www.rfc-editor.org/rfc/rfc9807.html

# Appendix A. Example session records

## A.1 Metric descriptor

```json
{  
"metric_id": "gpu.nvidia.memory_activity_pct",  
"display_name": "GPU global-memory activity",  
"entity_kind": "gpu",  
"unit": "percent",  
"value_kind": "gauge",  
"temporal_semantics": "vendor_sampled",  
"provider": "nvml",  
"source_semantics": "percent of vendor sample period during which global device memory was read or written",  
"comparability_group": "nvml_memory_time_busy",  
"capability_status": "available",  
"semantics_version": 2  
}
```

## A.2 Metric sample

```csv
# Native gpu stream (wide example)  
mono_ns,window_start_mono_ns,entity_id,gpu_util_pct,memory_activity_pct,vram_used_bytes,temp_c  
1234567890123,1234566890123,gpu:uuid:GPU-...,73,42,7340032000,67
```

## A.3 Application span events

```jsonl
{"timestamp_ns":1234567000000,"clock_domain":"linux_clock_monotonic","boot_id":"...","source":"iris","trace_id":"run-184","span_id":"path-3",  
"category":"compute","name":"PATH_SIMULATION","kind":"span_begin","attributes":{"batch":3,"backend":"cuda"},"sequence":41}  
{"timestamp_ns":1234575400000,"clock_domain":"linux_clock_monotonic","boot_id":"...","source":"iris","trace_id":"run-184","span_id":"path-3",  
"category":"compute","name":"PATH_SIMULATION","kind":"span_end","attributes":{"batch":3,"backend":"cuda"},"sequence":42}
```

# Appendix B. Capability examples

| Scenario                               | Expected SIA behavior                                                                                                                                                                                             |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NVIDIA discrete GPU + NVML             | Primary verified target on available hardware: show supported NVIDIA metrics; record UUID/PCI; authoritative benchmarks use target-local headless recording; native remote GUI is available for live observation. |
| Intel laptop iGPU without NVIDIA       | If no Intel GPU hardware is available, provider behavior may be fixture-tested only and remains hardware-unverified. CPU/Linux metrics remain independently supported on the tested Intel CPU host.               |
| AMD RX-class GPU                       | Implement capability discovery/DRM-hwmon parsing if desired, but because no AMD GPU is available for this project, label the provider hardware-unverified until a real-device session/test is supplied.           |
| No supported GPU                       | SIA remains a CPU/memory/I/O/system profiler; GPU panel may say 'No supported GPU metrics' rather than plotting zero.                                                                                             |
| Permission-denied process/GPU counter  | Preserve other metrics; mark capability permission_denied and show remediation in `sia doctor`.                                                                                                                 |
| Multiple NVIDIA GPUs                   | Separate UUID/PCI identities and traces; target process attribution is per device where supported.                                                                                                                |
| Imported custom/process-relative clock | Require synchronization anchors to SIA clock for precise alignment; otherwise label approximate and prohibit fine-grained causality claims.                                                                       |

# Appendix C. Architecture decisions

| ADR         | Decision                                                                                              | Rationale                                                                                                                                                            |
|-------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ADR-SIA-001 | Headless capture is authoritative for GPU profiling                                                   | Avoid GUI/wgpu observer effect and make capture reproducible.                                                                                                        |
| ADR-SIA-002 | One executable; modules, not services                                                                 | SIA is a local utility. No daemon/database architecture until proven necessary.                                                                                      |
| ADR-SIA-003 | Actual monotonic timestamps                                                                           | Required for application/system correlation and honest sampling jitter.                                                                                              |
| ADR-SIA-004 | Capability-driven metrics                                                                             | Fixes dead/false GPU traces and enables cross-vendor partial support.                                                                                                |
| ADR-SIA-005 | Buffered CSV/JSONL session store first                                                                | Meets transparency/export needs with low complexity; optimize only after evidence.                                                                                   |
| ADR-SIA-006 | Offline IRIS trace import is the primary integration                                                  | Preserves IRIS independence and cannot perturb/block production.                                                                                                     |
| ADR-SIA-007 | Optional Unix datagram markers                                                                        | Low overhead/nonblocking generic live annotations; loss is detectable.                                                                                               |
| ADR-SIA-008 | No Nsight/perf reimplementation                                                                       | SIA diagnoses whole-system bottleneck class; deep tools diagnose kernels/instructions.                                                                               |
| ADR-SIA-009 | Transparent diagnostics, no composite score                                                           | Evidence is inspectable; avoids false precision and opaque heuristics.                                                                                               |
| ADR-SIA-010 | Explicit clock domains + synchronization anchors                                                      | A monotonic number without its clock identity is not enough to align independent traces.                                                                             |
| ADR-SIA-011 | Wide append-only collector streams; long-form export                                                  | Reduces recorder volume/observer effect while retaining transparent files and easy analysis.                                                                         |
| ADR-SIA-012 | Progressive-disclosure information design                                                             | Overview, synchronized focus/context and details-on-demand preserve SIA clarity as metric count grows.                                                               |
| ADR-SIA-013 | Automatic diagnostics are non-authoritative                                                           | Raw aligned measurements are primary; diagnostics use inspectable claim levels and may be disabled.                                                                  |
| ADR-SIA-014 | Headless means no graphics initialization                                                             | Authoritative GPU captures must not create their own graphics workload.                                                                                              |
| ADR-SIA-015 | Prefer source-native counters/sample buffers                                                          | Higher polling rates cannot exceed the source's own resolution and may add observer cost.                                                                            |
| ADR-SIA-016 | IRIS JSONL import is the minimum mandatory adapter                                                    | Event traces are low-volume; avoid a Parquet dependency until evidence requires it.                                                                                  |
| ADR-SIA-018 | Off-target observer uses a minimal encrypted read-only SIA stream                                     | Measured machine stays headless; normal operator workflow is target `sia -headless`, observer exact printed `sia -o <IP> <phrase>` command.                  |
| ADR-SIA-019 | Observer live viewing is exploratory by default                                                       | Serialization/network work can perturb the target; authoritative performance claims use target-local headless capture unless equivalence is measured.                |
| ADR-SIA-020 | Verification is split into code, AI-agent and human classes                                           | Prevents subjective gates being falsely automated and avoids wasting human review on deterministic invariants.                                                       |
| ADR-SIA-021 | Unavailable GPU vendors may be implemented but not certified                                          | Fixtures can validate parsing/contracts, not real hardware semantics or observer effect.                                                                             |
| ADR-SIA-022 | Three-word phrase authenticates a PAKE; it is not an encryption key                                   | Human-friendly session credential plus standard PAKE gives strong ephemeral session keys without long tokens, certificates/accounts or home-made low-entropy crypto. |
| ADR-SIA-023 | Encrypted observer transport is measured, not assumed free                                            | Authoritative recording remains local; observer encryption/network cost gets its own equivalence benchmark.                                                          |
| ADR-SIA-024 | Normal 0.1 observer transport has no plaintext fallback                                               | Avoids accidental leakage and divergent code paths; debugging insecure transport cannot satisfy release tests.                                                       |
| ADR-SIA-025 | Pairing phrase is ephemeral and not persisted                                                         | No credential database or certificate lifecycle; each headless run establishes a new pairing context.                                                                |
| ADR-SIA-026 | Phrase strength is derived from bounded online-guess risk, not a hard-coded bit count                 | PAKE removes offline guessing; phrase-space size should follow the actual attempt/threat policy while human vocabulary quality is reviewed separately.               |
| ADR-SIA-027 | Full phrase is printed in the default observer command despite observer-shell-history exposure        | Credential is session-ephemeral and local observer users are in the trust boundary; optional prompt mode can avoid shell history without degrading default UX.       |
| ADR-SIA-028 | One active observer; same phrase permits reconnect for the session                                    | Meets the use case with minimum server complexity and predictable observer-effect load.                                                                              |
| ADR-SIA-029 | PAKE protocol is normative by security properties; implementation library is selected by security ADR | Avoid hard-wiring an unaudited dependency into the architecture while prohibiting home-grown crypto; operator UX remains stable.                                     |
| ADR-SIA-030 | No live phrase-regeneration control in 0.1                                                            | One session phrase + restart-on-compromise is simpler and avoids adding a remote credential-management path.                                                         |
