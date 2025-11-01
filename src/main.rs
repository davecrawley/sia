use eframe::{egui, egui::Vec2};
use egui::{Align2, Color32, FontFamily, FontId, RichText, TextStyle};
use egui_plot::{Corner, Legend, Line, Plot, PlotBounds, PlotPoints, Text};
use once_cell::sync::Lazy;
use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::io::Read;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use sysinfo::{CpuExt, System, SystemExt};

// ===================== Optional NVIDIA support =====================
#[cfg(feature = "nvidia")]
mod nvgpu {
    use nvml_wrapper::{
        enum_wrappers::device::{Clock as NvClock, TemperatureSensor},
        Nvml,
    };

    pub struct NvState {
        pub nvml: Nvml,
        pub device_index: u32,
    }

    impl NvState {
        pub fn try_new() -> Option<Self> {
            let nvml = Nvml::init().ok()?;
            let count = nvml.device_count().ok()?;
            if count == 0 { return None; }
            let idx = 0u32;
            let _ = nvml.device_by_index(idx).ok()?; // probe
            Some(Self { nvml, device_index: idx })
        }
    }

    pub fn first_gpu_metrics(state: &NvState) -> Option<(f64, f64, f64)> {
        let dev = state.nvml.device_by_index(state.device_index).ok()?;
        let util = dev.utilization_rates().ok()?; // gpu, mem (% u32)
        let mem = dev.memory_info().ok()?; // bytes
        let temp = dev.temperature(TemperatureSensor::Gpu).ok()? as f64; // °C
        let mem_pct = if mem.total > 0 { (mem.used as f64 / mem.total as f64) * 100.0 } else { 0.0 };
        Some((util.gpu as f64, mem_pct, temp))
    }

    /// Returns clocks in MHz: (graphics, sm, memory, video)
    pub fn gpu_clocks_mhz(state: &NvState) -> Option<(f64, f64, f64, f64)> {
        let dev = state.nvml.device_by_index(state.device_index).ok()?;
        let g = dev.clock_info(NvClock::Graphics).ok()? as f64;
        let sm = dev.clock_info(NvClock::SM).ok().map(|v| v as f64).unwrap_or(g);
        let m = dev.clock_info(NvClock::Memory).ok()? as f64;
        let v = dev.clock_info(NvClock::Video).ok().map(|v| v as f64).unwrap_or(g);
        Some((g, sm, m, v))
    }
}

// ===================== Time series helpers =====================
#[derive(Default, Clone)]
struct RollingSeries {
    xs: VecDeque<f64>,
    ys: VecDeque<f64>,
    cap: usize,
}
impl RollingSeries {
    fn new(cap: usize) -> Self { Self { xs: VecDeque::with_capacity(cap), ys: VecDeque::with_capacity(cap), cap } }
    fn push(&mut self, x: f64, y: f64) { if self.xs.len() == self.cap { self.xs.pop_front(); self.ys.pop_front(); } self.xs.push_back(x); self.ys.push_back(y); }
    fn points_after(&self, x_min: f64) -> PlotPoints {
        let mut out: Vec<[f64; 2]> = Vec::with_capacity(self.xs.len());
        for (x, y) in self.xs.iter().zip(self.ys.iter()) { if *x >= x_min { out.push([*x, *y]); } }
        PlotPoints::from(out)
    }
    fn points_after_scaled(&self, x_min: f64, div: f64) -> PlotPoints {
        let mut out: Vec<[f64; 2]> = Vec::with_capacity(self.xs.len());
        for (x, y) in self.xs.iter().zip(self.ys.iter()) { if *x >= x_min { out.push([*x, *y / div]); } }
        PlotPoints::from(out)
    }
    fn min_max_y(&self, x_min: f64, x_max: f64) -> Option<(f64, f64)> {
        let mut mn = f64::INFINITY; let mut mx = f64::NEG_INFINITY;
        for (x, y) in self.xs.iter().zip(self.ys.iter()) {
            if *x >= x_min && *x <= x_max { if *y < mn { mn = *y; } if *y > mx { mx = *y; } }
        }
        if mn.is_finite() && mx.is_finite() { Some((mn, mx)) } else { None }
    }
    fn last_y(&self) -> Option<f64> { self.ys.back().copied() }
}

// ===================== Sensors discovery =====================
#[derive(Clone, Debug)]
struct TempSensor { raw_name: String, raw_label: String, path: PathBuf }
static HWMON_SENSORS: Lazy<Vec<TempSensor>> = Lazy::new(discover_hwmon_temps);

#[derive(Clone, Debug)]
struct FreqSensor { core: usize, path: PathBuf }
static FREQ_SENSORS: Lazy<Vec<FreqSensor>> = Lazy::new(discover_cpu_freqs);

fn discover_cpu_freqs() -> Vec<FreqSensor> {
    let mut sensors = vec![];
    if let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") {
        for e in entries.flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if !name.starts_with("cpu") { continue; }
            let idx = match name.trim_start_matches("cpu").parse::<usize>() { Ok(v)=>v, Err(_)=>continue };
            let cf = p.join("cpufreq");
            let cand1 = cf.join("scaling_cur_freq");
            let cand2 = cf.join("cpuinfo_cur_freq");
            let path = if cand1.exists() { cand1 } else if cand2.exists() { cand2 } else { continue };
            sensors.push(FreqSensor { core: idx, path });
        }
    }
    sensors.sort_by_key(|s| s.core);
    sensors
}

fn read_freq_khz(path: &PathBuf) -> Option<f64> { let mut s=String::new(); fs::File::open(path).ok()?.read_to_string(&mut s).ok()?; s.trim().parse::<f64>().ok() }

fn discover_hwmon_temps() -> Vec<TempSensor> {
    let mut sensors = vec![];
    if let Ok(entries) = fs::read_dir("/sys/class/hwmon") {
        for e in entries.flatten() {
            let base = e.path();
            let name = fs::read_to_string(base.join("name")).unwrap_or_default().trim().to_string();
            if let Ok(files) = fs::read_dir(&base) {
                for f in files.flatten() {
                    let p = f.path(); let fname = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if fname.starts_with("temp") && fname.ends_with("_input") {
                        let mut label = name.clone();
                        let label_path = base.join(fname.replace("_input","_label"));
                        if let Ok(lbl) = fs::read_to_string(label_path) { let l=lbl.trim(); if !l.is_empty() { label = l.to_string(); } }
                        sensors.push(TempSensor { raw_name: name.clone(), raw_label: label, path: p.clone() });
                    }
                }
            }
        }
    }
    sensors
}

fn read_temp_c(path: &PathBuf) -> Option<f64> { let mut s=String::new(); fs::File::open(path).ok()?.read_to_string(&mut s).ok()?; let v: f64 = s.trim().parse().ok()?; Some(if v>1000.0 { v/1000.0 } else { v }) }

// ===================== Grouping & taxonomy =====================
#[derive(Clone, Debug)]
struct SensorItem { name: String, idx: usize, visible: bool, color: Color32 }
#[derive(Clone, Debug)]
struct SensorGroup { key: String, display: String, items: Vec<SensorItem>, visible: bool, warn: f64, hot: f64, show_thresholds: bool }

fn classify(raw: &str) -> (String, String, f64, f64) {
    let r = raw.to_lowercase();
    if r.contains("coretemp") || r.contains("k10temp") || r.contains("zen") || r.contains("cpu") { return ("cpu".into(), "CPU".into(), 90.0, 100.0); }
    if r.contains("amdgpu") { return ("gpu".into(), "GPU (amdgpu)".into(), 85.0, 95.0); }
    if r.contains("nvidia") || r.contains("gpu") { return ("gpu".into(), "GPU (nvidia)".into(), 85.0, 95.0); }
    if r.contains("nvme") { return ("nvme".into(), "NVMe SSD".into(), 70.0, 80.0); }
    if r.contains("spd") { return ("ramspd".into(), "Memory (SPD Hub)".into(), 70.0, 85.0); }
    if r.contains("iwlwifi") { return ("wifi".into(), "Wi‑Fi Controller (iwlwifi)".into(), 80.0, 90.0); }
    if r.contains("r8169") { return ("eth".into(), "Ethernet Controller (r8169)".into(), 80.0, 90.0); }
    if r.contains("igc") { return ("eth".into(), "Ethernet Controller (igc)".into(), 80.0, 90.0); }
    if r.contains("e1000") { return ("eth".into(), "Ethernet Controller (e1000)".into(), 80.0, 90.0); }
    if r.contains("r8125") { return ("eth".into(), "Ethernet Controller (r8125)".into(), 80.0, 90.0); }
    if r.contains("acpitz") { return ("acpi".into(), "System Temperature (acpitz)".into(), 80.0, 95.0); }
    if r.contains("pch") || r.contains("isa") { return ("chipset".into(), "Chipset".into(), 85.0, 95.0); }
    (raw.to_string(), raw.to_string(), 90.0, 100.0)
}

fn nice_label(group: &str, raw: &str) -> String {
    let ll = raw.to_lowercase();
    match group {
        "CPU" => { if ll.contains("package") { "CPU (Package)".into() } else if ll.contains("tctl") || ll.contains("tdie") { "CPU (Composite)".into() } else if ll.starts_with("core ") { raw.replace("Core ", "CPU (Core ") + ")" } else { raw.into() } }
        "GPU (amdgpu)" | "GPU (nvidia)" => { if ll.contains("edge") { "GPU (Edge)".into() } else if ll.contains("hotspot") { "GPU (Hotspot)".into() } else { raw.into() } }
        "NVMe SSD" => raw.replace("Composite","SSD"),
        _ => raw.into(),
    }
}

fn palette() -> Vec<Color32> { vec![ Color32::from_rgb(244,67,54), Color32::from_rgb(33,150,243), Color32::from_rgb(76,175,80), Color32::from_rgb(255,152,0), Color32::from_rgb(156,39,176), Color32::from_rgb(121,85,72), Color32::from_rgb(63,81,181), Color32::from_rgb(0,150,136), Color32::from_rgb(205,220,57), Color32::from_rgb(233,30,99), Color32::from_rgb(158,158,158), Color32::from_rgb(255,87,34), Color32::from_rgb(3,169,244), Color32::from_rgb(139,195,74), Color32::from_rgb(171,71,188), Color32::from_rgb(255,238,88), Color32::from_rgb(38,198,218), Color32::from_rgb(141,110,99), Color32::from_rgb(120,144,156) ] }

fn theme_color(kind: &str) -> Color32 {
    match kind {
        "cpu" => Color32::from_rgb(244,67,54),     // red
        "gpu" => Color32::from_rgb(33,150,243),    // blue
        "nvme" => Color32::from_rgb(255,152,0),    // orange
        "ramspd" => Color32::from_rgb(76,175,80),  // green
        "wifi" => Color32::from_rgb(0,150,136),    // teal
        "eth" => Color32::from_rgb(171,71,188),    // purple
        _ => Color32::from_rgb(158,158,158),        // grey
    }
}
fn tint(c: Color32, factor: f32) -> Color32 {
    let (r,g,b,a) = (c.r() as f32, c.g() as f32, c.b() as f32, c.a());
    let t = |v: f32| -> u8 { v.min(255.0).max(0.0) as u8 };
    Color32::from_rgba_unmultiplied(t(r + (255.0-r)*factor), t(g + (255.0-g)*factor), t(b + (255.0-b)*factor), a)
}

fn build_groups() -> Vec<SensorGroup> {
    let mut map: BTreeMap<String, SensorGroup> = BTreeMap::new();
    for (idx, s) in HWMON_SENSORS.iter().enumerate() {
        let (key, display, warn, hot) = classify(&s.raw_name);
        let entry = map.entry(key.clone()).or_insert(SensorGroup {
            key: key.clone(),
            display: display.clone(),
            items: vec![],
            visible: true,
            warn,
            hot,
            show_thresholds: false,
        });
        let label = nice_label(&display, &s.raw_label);
        entry.items.push(SensorItem { name: label, idx, visible: false, color: Color32::WHITE });
    }

    // defaults: prefer composite/package/system/wifi/ethernet
    for g in map.values_mut() {
        let mut showed = false;
        for it in &mut g.items {
            let n = it.name.to_lowercase();
            let is_composite = n.contains("composite") || n.contains("package") || n.contains("core)");
            if !showed && (is_composite || g.display.contains("Wi‑Fi Controller") || g.display.contains("Ethernet Controller") || g.display.contains("System Temperature")) {
                it.visible = true;
                showed = true;
            }
        }
        if !showed { if let Some(first) = g.items.first_mut() { first.visible = true; } }
    }

    // assign colors per group with gentle tints so the same thing stays the same color across plots
    for g in map.values_mut() {
        let base = theme_color(&g.key);
        for (i, it) in g.items.iter_mut().enumerate() {
            it.color = tint(base, (i as f32) * 0.08);
        }
    }

    // sort items within groups
    for g in map.values_mut() {
        if g.display.starts_with("CPU") {
            g.items.sort_by(|a, b| {
                fn key(name: &str) -> (u8, i32, String) {
                    let ln = name.to_lowercase();
                    let mut tier: u8 = 3;
                    if ln.contains("package") || ln.contains("composite") { tier = 0; }
                    else if ln.contains("cpu (core ") { tier = 1; }
                    let mut idx: i32 = i32::MAX;
                    if let Some(start) = ln.find("cpu (core ") {
                        if let Some(end) = ln[start+11..].find(')') {
                            let num = &ln[start+11..start+11+end];
                            idx = num.parse::<i32>().unwrap_or(i32::MAX);
                        }
                    }
                    (tier, idx, ln)
                }
                key(&a.name).cmp(&key(&b.name))
            });
        } else if g.display.starts_with("GPU") {
            g.items.sort_by(|a, b| {
                fn tier(name: &str) -> u8 {
                    let n = name.to_lowercase();
                    if n.contains("edge") { 0 } else if n.contains("hotspot") { 1 } else { 2 }
                }
                tier(&a.name).cmp(&tier(&b.name)).then(a.name.cmp(&b.name))
            });
        } else {
            g.items.sort_by(|a, b| a.name.cmp(&b.name));
        }
    }

    // collect & order groups: CPU, GPU, NVMe SSD, Memory (SPD), Wi‑Fi, Ethernet, others
    let mut v: Vec<_> = map.into_values().collect();
    fn rank(key: &str) -> i32 { match key { "cpu"=>0, "gpu"=>1, "nvme"=>2, "ramspd"=>3, "wifi"=>4, "eth"=>5, _=>6 } }
    v.sort_by_key(|g| rank(&g.key));
    v
}

// ===================== App model =====================
#[derive(Clone, Copy, PartialEq, Eq)]
enum LegendPlacement { Footer, Side }

struct App {
    // meta
    start: Instant,
    sys: System,

    // utilization series
    cpu_util: RollingSeries,
    ram_util: RollingSeries,
    gpu_util: RollingSeries,
    vram_util: RollingSeries,

    // temps & freq
    temp_series: Vec<RollingSeries>,
    freq_series: Vec<RollingSeries>,       // CPU core kHz
    freq_visible: Vec<bool>,
    freq_colors: Vec<Color32>,

    // sensor groups
    groups: Vec<SensorGroup>,

    // sampling
    seconds: f64,
    sample_period: Duration,
    last_tick: Instant,

    // UI state
    display_window_secs: f64,
    legend_place: LegendPlacement,
    ui_font_size: f32,
    ui_font_color: Color32,
    pending_ui_font_size: f32,
    pending_ui_font_color: Color32,
    live_font_preview: bool,

    // NVIDIA (optional)
    #[cfg(feature = "nvidia")]
    nv: Option<nvgpu::NvState>,
    #[cfg(feature = "nvidia")]
    gpu_temp_idx: Option<usize>,
    #[cfg(feature = "nvidia")]
    gpu_clk_graphics: RollingSeries,   // MHz
    #[cfg(feature = "nvidia")]
    gpu_clk_sm: RollingSeries,         // MHz
    #[cfg(feature = "nvidia")]
    gpu_clk_mem: RollingSeries,        // MHz
    #[cfg(feature = "nvidia")]
    gpu_clk_video: RollingSeries,      // MHz
    #[cfg(feature = "nvidia")]
    gpu_freq_graphics_vis: bool,
    #[cfg(feature = "nvidia")]
    gpu_freq_sm_vis: bool,
    #[cfg(feature = "nvidia")]
    gpu_freq_mem_vis: bool,
    #[cfg(feature = "nvidia")]
    gpu_freq_video_vis: bool,
    #[cfg(feature = "nvidia")]
    gpu_mem_effective: bool,
}

impl App {
    fn new(capacity_secs: usize, sample_hz: f64) -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let groups = build_groups();
        let temp_series = HWMON_SENSORS.iter().map(|_| RollingSeries::new(capacity_secs)).collect::<Vec<_>>();
        let freq_series = FREQ_SENSORS.iter().map(|_| RollingSeries::new(capacity_secs)).collect::<Vec<_>>();
        let freq_visible = FREQ_SENSORS.iter().map(|_| true).collect::<Vec<_>>();
        let pal = palette();
        let mut freq_colors = Vec::with_capacity(FREQ_SENSORS.len());
        for i in 0..FREQ_SENSORS.len() { freq_colors.push(pal[i % pal.len()]); }

        #[cfg(feature = "nvidia")]
        let nv_opt = nvgpu::NvState::try_new();
        #[cfg(feature = "nvidia")]
        let (mut temp_series, mut groups, gpu_temp_idx_opt) = {
            let mut temp_series = temp_series;
            let mut groups = groups;
            let idx = temp_series.len();
            temp_series.push(RollingSeries::new(capacity_secs));
            // Ensure GPU group exists and add synthetic GPU temp line
            if let Some(g) = groups.iter_mut().find(|g| g.display.starts_with("GPU")) {
                g.items.push(SensorItem { name: "GPU (Core)".into(), idx, visible: true, color: Color32::WHITE });
            } else {
                let mut g = SensorGroup { key: "gpu".into(), display: "GPU".into(), items: vec![], visible: true, warn: 85.0, hot: 95.0, show_thresholds: false };
                g.items.push(SensorItem { name: "GPU (Core)".into(), idx, visible: true, color: Color32::WHITE });
                groups.push(g);
            }
            // keep GPU sorted after CPU
            fn rank(key: &str) -> i32 { match key { "cpu"=>0, "gpu"=>1, "nvme"=>2, "ramspd"=>3, "wifi"=>4, "eth"=>5, _=>6 } }
            groups.sort_by_key(|g| rank(&g.key));
            (temp_series, groups, Some(idx))
        };
        #[cfg(not(feature = "nvidia"))]
        let (nv_opt, gpu_temp_idx_opt) = (None::<()> as Option<()>, None::<usize>);

        Self {
            start: Instant::now(),
            sys,
            cpu_util: RollingSeries::new(capacity_secs),
            ram_util: RollingSeries::new(capacity_secs),
            gpu_util: RollingSeries::new(capacity_secs),
            vram_util: RollingSeries::new(capacity_secs),
            temp_series,
            freq_series,
            freq_visible,
            freq_colors,
            groups,
            seconds: 0.0,
            sample_period: Duration::from_secs_f64((1.0 / sample_hz).max(0.05)),
            last_tick: Instant::now(),
            display_window_secs: 120.0,
            legend_place: LegendPlacement::Footer,
            ui_font_size: 14.0,
            ui_font_color: Color32::WHITE,
            pending_ui_font_size: 14.0,
            pending_ui_font_color: Color32::WHITE,
            live_font_preview: false,
            #[cfg(feature = "nvidia")]
            nv: nv_opt,
            #[cfg(feature = "nvidia")]
            gpu_temp_idx: gpu_temp_idx_opt,
            #[cfg(feature = "nvidia")]
            gpu_clk_graphics: RollingSeries::new(capacity_secs),
            #[cfg(feature = "nvidia")]
            gpu_clk_sm: RollingSeries::new(capacity_secs),
            #[cfg(feature = "nvidia")]
            gpu_clk_mem: RollingSeries::new(capacity_secs),
            #[cfg(feature = "nvidia")]
            gpu_clk_video: RollingSeries::new(capacity_secs),
            #[cfg(feature = "nvidia")]
            gpu_freq_graphics_vis: true,
            #[cfg(feature = "nvidia")]
            gpu_freq_sm_vis: true,
            #[cfg(feature = "nvidia")]
            gpu_freq_mem_vis: true,
            #[cfg(feature = "nvidia")]
            gpu_freq_video_vis: false,
            #[cfg(feature = "nvidia")]
            gpu_mem_effective: false,
        }
    }

    fn sample(&mut self) {
        // refresh
        self.sys.refresh_cpu();
        self.sys.refresh_memory();

        // CPU / RAM %
        let avg_cpu: f32 = self.sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / (self.sys.cpus().len().max(1) as f32);
        let cpu_pct = avg_cpu as f64;
        let total = self.sys.total_memory() as f64;
        let used = self.sys.used_memory() as f64;
        let ram_pct = if total>0.0 { (used/total)*100.0 } else { 0.0 };

        // timebase
        self.seconds += self.sample_period.as_secs_f64();

        // push util
        self.cpu_util.push(self.seconds, cpu_pct);
        self.ram_util.push(self.seconds, ram_pct);

        // NVIDIA sampling
        #[cfg(feature = "nvidia")]
        {
            if let Some(nv) = &self.nv {
                if let Some((gpu_pct, vram_pct, temp_c)) = nvgpu::first_gpu_metrics(nv) {
                    self.gpu_util.push(self.seconds, gpu_pct);
                    self.vram_util.push(self.seconds, vram_pct);
                    if let Some(idx) = self.gpu_temp_idx { self.temp_series[idx].push(self.seconds, temp_c); }
                } else {
                    self.gpu_util.push(self.seconds, f64::NAN);
                    self.vram_util.push(self.seconds, f64::NAN);
                }
                if let Some((g, sm, m, v)) = nvgpu::gpu_clocks_mhz(nv) {
                    self.gpu_clk_graphics.push(self.seconds, g);
                    self.gpu_clk_sm.push(self.seconds, sm);
                    self.gpu_clk_mem.push(self.seconds, m);
                    self.gpu_clk_video.push(self.seconds, v);
                }
            } else {
                self.gpu_util.push(self.seconds, f64::NAN);
                self.vram_util.push(self.seconds, f64::NAN);
            }
        }
        #[cfg(not(feature = "nvidia"))]
        {
            self.gpu_util.push(self.seconds, f64::NAN);
            self.vram_util.push(self.seconds, f64::NAN);
        }

        // CPU per-core frequencies (kHz)
        for (i, fsens) in FREQ_SENSORS.iter().enumerate() {
            if let Some(khz) = read_freq_khz(&fsens.path) { self.freq_series[i].push(self.seconds, khz); }
        }
        // Temperatures
        for (i, ts) in HWMON_SENSORS.iter().enumerate() {
            if let Some(t) = read_temp_c(&ts.path) { self.temp_series[i].push(self.seconds, t); }
        }
    }
}

// ===================== UI =====================
impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Typography
        let mut style: egui::Style = (*ctx.style()).clone();
        style.visuals.override_text_color = Some(self.ui_font_color);
        style.text_styles = [
            (TextStyle::Heading,  FontId::new(self.ui_font_size, FontFamily::Proportional)),
            (TextStyle::Body,     FontId::new(self.ui_font_size, FontFamily::Proportional)),
            (TextStyle::Monospace,FontId::new(self.ui_font_size, FontFamily::Monospace)),
            (TextStyle::Button,   FontId::new(self.ui_font_size, FontFamily::Proportional)),
            (TextStyle::Small,    FontId::new(self.ui_font_size, FontFamily::Proportional)),
        ].into();
        ctx.set_style(style);

        // sampling
        if self.last_tick.elapsed() >= self.sample_period { self.sample(); self.last_tick = Instant::now(); }
        ctx.request_repaint_after(Duration::from_millis(16));

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SIA - System Information Analyzer - by David Crawley dc@ubiquityrobotics.com");
                ui.separator();
                ui.label(format!("Uptime: {}s", self.start.elapsed().as_secs()));
                ui.separator();
                ui.label(format!("Samples: {}", (self.seconds / self.sample_period.as_secs_f64()) as usize));
                ui.separator();
                ui.label(format!("CPU: {:.0}%", self.cpu_util.last_y().unwrap_or(0.0)));
                ui.separator();
                ui.label(format!("RAM: {:.0}%", self.ram_util.last_y().unwrap_or(0.0)));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.set_min_size(Vec2::new(980.0, 640.0));
            let (auto_xmin, auto_xmax) = if self.seconds > self.display_window_secs { (self.seconds - self.display_window_secs, self.seconds) } else { (0.0, self.display_window_secs) };

            // ============ Utilization ============
            ui.heading("Utilization");
            let util_plot = Plot::new("util").height(220.0).allow_scroll(true).allow_zoom(true).legend(Legend::default().position(Corner::LeftTop));
            util_plot.show(ui, |plot_ui| {
                let (xmin, xmax) = (auto_xmin, auto_xmax);
                plot_ui.set_plot_bounds(PlotBounds::from_min_max([xmin, 0.0], [xmax, 100.0]));
                // left-side labels in UI color
                let (ymin, ymax) = (0.0, 100.0);
                let ticks = 4; let step = (ymax - ymin) / (ticks as f64); let mut v = ymin;
                while v <= ymax + 1e-6 { plot_ui.text(Text::new([xmin, v].into(), format!("{:.0}%", v)).anchor(Align2::LEFT_CENTER)); v += step; }

                plot_ui.line(Line::new(self.cpu_util.points_after(xmin)).name("CPU %").color(theme_color("cpu")));
                plot_ui.line(Line::new(self.gpu_util.points_after(xmin)).name("GPU %").color(theme_color("gpu")));
                plot_ui.line(Line::new(self.ram_util.points_after(xmin)).name("RAM %").color(theme_color("ramspd")));
                plot_ui.line(Line::new(self.vram_util.points_after(xmin)).name("VRAM %").color(theme_color("nvme")));

                // right-side labels for symmetry
                let mut v2 = ymin; while v2 <= ymax + 1e-6 { plot_ui.text(Text::new([xmax, v2].into(), format!("{:.0}%", v2)).anchor(Align2::RIGHT_CENTER)); v2 += step; }
            });

            ui.separator();

            // ============ Temperatures ============
            ui.heading("Temperatures (°C)");
            let (xmin, xmax) = (auto_xmin, auto_xmax);
            let temp_plot = Plot::new("temps").height(260.0).allow_scroll(true).allow_zoom(true);
            temp_plot.show(ui, |plot_ui| {
                // dynamic y
                let mut mn = f64::INFINITY; let mut mx = f64::NEG_INFINITY;
                for g in &self.groups { if !g.visible { continue; } for it in &g.items { if !it.visible { continue; } if let Some((a,b)) = self.temp_series[it.idx].min_max_y(xmin, xmax) { if a < mn { mn = a; } if b > mx { mx = b; } } }}
                if !mn.is_finite() || !mx.is_finite() || (mx - mn).abs() < 1e-6 { mn = 0.0; mx = 120.0; }
                let pad = ((mx - mn) * 0.1).max(2.0); mn = (mn - pad).max(0.0); mx = (mx + pad).min(130.0);
                plot_ui.set_plot_bounds(PlotBounds::from_min_max([xmin, mn], [xmax, mx]));

                for g in &self.groups { if !g.visible { continue; }
                    for it in &g.items { if !it.visible { continue; }
                        let pts = self.temp_series[it.idx].points_after(xmin);
                        plot_ui.line(Line::new(pts).name(format!("{}: {}", g.display, it.name)).color(it.color));
                    }
                }
                let ticks = 4; let step = (mx - mn) / (ticks as f64); let mut v = mn;
                while v <= mx + 1e-6 { plot_ui.text(Text::new([xmax, v].into(), format!("{:.0}", v)).anchor(Align2::RIGHT_CENTER)); v += step; }
            });

            ui.separator();

            // ============ Frequencies (GHz) ============
            ui.heading("Frequencies (GHz)");
            let (xmin, xmax) = (auto_xmin, auto_xmax);
            let freq_plot = Plot::new("freq").height(240.0).allow_scroll(true).allow_zoom(true);
            freq_plot.show(ui, |plot_ui| {
                // dynamic y across CPU cores + GPU lines
                let mut mn = f64::INFINITY; let mut mx = f64::NEG_INFINITY;
                for (i, series) in self.freq_series.iter().enumerate() {
                    if !self.freq_visible.get(i).copied().unwrap_or(false) { continue; }
                    if let Some((a,b)) = series.min_max_y(xmin, xmax) { let ag=a/1_000_000.0; let bg=b/1_000_000.0; if ag<mn{mn=ag;} if bg>mx{mx=bg;} }
                }
                #[cfg(feature = "nvidia")]
                {
                    if self.gpu_freq_graphics_vis { if let Some((a,b)) = self.gpu_clk_graphics.min_max_y(xmin, xmax) { let ag=a/1000.0; let bg=b/1000.0; if ag<mn{mn=ag;} if bg>mx{mx=bg;} } }
                    if self.gpu_freq_sm_vis       { if let Some((a,b)) = self.gpu_clk_sm.min_max_y(xmin, xmax)       { let ag=a/1000.0; let bg=b/1000.0; if ag<mn{mn=ag;} if bg>mx{mx=bg;} } }
                    if self.gpu_freq_mem_vis      { if let Some((a,b)) = self.gpu_clk_mem.min_max_y(xmin, xmax)      { let ag=(a/1000.0) * if self.gpu_mem_effective { 2.0 } else { 1.0 }; let bg=(b/1000.0) * if self.gpu_mem_effective { 2.0 } else { 1.0 }; if ag<mn{mn=ag;} if bg>mx{mx=bg;} } }
                    if self.gpu_freq_video_vis    { if let Some((a,b)) = self.gpu_clk_video.min_max_y(xmin, xmax)    { let ag=a/1000.0; let bg=b/1000.0; if ag<mn{mn=ag;} if bg>mx{mx=bg;} } }
                }
                if !mn.is_finite() || !mx.is_finite() || (mx - mn).abs() < 1e-6 { mn = 0.1; mx = 10.0; }
                let pad = ((mx - mn) * 0.08).max(0.05); mn = (mn - pad).max(0.0); mx = (mx + pad).min(12.0);
                plot_ui.set_plot_bounds(PlotBounds::from_min_max([xmin, mn], [xmax, mx]));

                for (i, series) in self.freq_series.iter().enumerate() {
                    if !self.freq_visible.get(i).copied().unwrap_or(false) { continue; }
                    let name = format!("CPU Core {}", FREQ_SENSORS.get(i).map(|s| s.core).unwrap_or(i));
                    let pts = series.points_after_scaled(xmin, 1_000_000.0);
                    plot_ui.line(Line::new(pts).name(name).color(self.freq_colors[i % self.freq_colors.len()]));
                }
                #[cfg(feature = "nvidia")]
                {
                    if self.gpu_freq_graphics_vis { let pts = self.gpu_clk_graphics.points_after_scaled(xmin, 1000.0); plot_ui.line(Line::new(pts).name("GPU Graphics")); }
                    if self.gpu_freq_sm_vis       { let pts = self.gpu_clk_sm.points_after_scaled(xmin, 1000.0);       plot_ui.line(Line::new(pts).name("GPU SM")); }
                    if self.gpu_freq_mem_vis      { let div = 1000.0 / if self.gpu_mem_effective { 2.0 } else { 1.0 }; let pts = self.gpu_clk_mem.points_after_scaled(xmin, div); let label = if self.gpu_mem_effective { "GPU Memory (effective)" } else { "GPU Memory" }; plot_ui.line(Line::new(pts).name(label)); }
                    if self.gpu_freq_video_vis    { let pts = self.gpu_clk_video.points_after_scaled(xmin, 1000.0);    plot_ui.line(Line::new(pts).name("GPU Video")); }
                }
                let ticks = 4; let step = (mx - mn) / (ticks as f64); let mut v = mn;
                while v <= mx + 1e-6 { plot_ui.text(Text::new([xmax, v].into(), format!("{:.2} GHz", v)).anchor(Align2::RIGHT_CENTER)); v += step; }
            });

            // ============ Legends outside ============
            match self.legend_place { LegendPlacement::Footer => self.footer_legend(ui), LegendPlacement::Side => self.side_legend(ui) }

            ui.separator();

            // ============ Settings & Sensors ============
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                ui.heading("Display");
                ui.horizontal(|ui| {
                    ui.label("Window (seconds) before scroll):");
                    ui.add(egui::Slider::new(&mut self.display_window_secs, 30.0..=900.0));
                    egui::ComboBox::from_label("Legend placement")
                        .selected_text(match self.legend_place { LegendPlacement::Footer => "Footer", LegendPlacement::Side => "Side" })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.legend_place, LegendPlacement::Footer, "Footer");
                            ui.selectable_value(&mut self.legend_place, LegendPlacement::Side, "Side strip");
                        });
                    ui.separator();
                    ui.label("Font size");
                    let resp = ui.add(egui::Slider::new(&mut self.pending_ui_font_size, 10.0..=22.0));
                    if self.live_font_preview { self.ui_font_size = self.pending_ui_font_size; }
                    if resp.drag_stopped() { self.ui_font_size = self.pending_ui_font_size; }
                    ui.label("Font color");
                    let _cresp = ui.color_edit_button_srgba(&mut self.pending_ui_font_color);
                    if self.live_font_preview { self.ui_font_color = self.pending_ui_font_color; }
                    ui.separator();
                    if ui.button("Apply font").clicked() { self.ui_font_size = self.pending_ui_font_size; self.ui_font_color = self.pending_ui_font_color; }
                    ui.toggle_value(&mut self.live_font_preview, "Live preview");
                });
                ui.separator();

                ui.heading("Sensors");
                let cols = 2;
                egui::Grid::new("sensor_grid").num_columns(cols).striped(true).min_col_width(400.0).spacing([18.0, 8.0]).show(ui, |ui| {
                    for g in &mut self.groups {
                        if g.display.starts_with("CPU") {
                            egui::CollapsingHeader::new("CPU").id_source("grp_cpu").default_open(false).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    //let avail = ui.available_width();
                                    //let min_right = 250.0;
                                    //let max_left  = (avail - min_right).max(0.0);
                                    //let target    = avail * 0.9;


                                    let left_px   = 150.0; //target.min(max_left);
                                    let right_px  = 250.0;  //(avail - left_px).max(0.0);
                                    let layout = egui::Layout::top_down(egui::Align::LEFT);
                                    ui.allocate_ui_with_layout(egui::vec2(left_px, 0.0), layout, |ui| {
                                        ui.label(RichText::new("Core temperatures").strong());
                                        for it in &mut g.items { ui.checkbox(&mut it.visible, &it.name); }
                                    });
                                    ui.allocate_ui_with_layout(egui::vec2(right_px, 0.0), layout, |ui| {
                                        ui.label(RichText::new("Core frequencies").strong());
                                        ui.horizontal(|ui| {
                                            if ui.button("All").clicked()  { for v in &mut self.freq_visible { *v = true; } }
                                            if ui.button("None").clicked() { for v in &mut self.freq_visible { *v = false; } }
                                        });
                                        for (i, fs) in FREQ_SENSORS.iter().enumerate() {
                                            let mut vis = self.freq_visible[i];
                                            let label = format!("CPU Core {}", fs.core);
                                            ui.checkbox(&mut vis, label);
                                            self.freq_visible[i] = vis;
                                        }
                                    });
                                });
                            });
                        } else if g.display.starts_with("GPU") {
                            egui::CollapsingHeader::new("GPU").id_source("grp_gpu").default_open(false).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    //let avail = ui.available_width();
                                    //let min_right = 250.0;
                                    //let max_left  = (avail - min_right).max(0.0);
                                    //let target    = avail * 0.9;
                                    let left_px   = 150.0; //target.min(max_left);
                                    let right_px  = 250.0; //(avail - left_px).max(0.0);
                                    let layout = egui::Layout::top_down(egui::Align::LEFT);
                                    ui.allocate_ui_with_layout(egui::vec2(left_px, 0.0), layout, |ui| {
                                        ui.label(RichText::new("Temperatures").strong());
                                        for it in &mut g.items { ui.checkbox(&mut it.visible, &it.name); }
                                    });
                                    ui.allocate_ui_with_layout(egui::vec2(right_px, 0.0), layout, |ui| {
                                        ui.label(RichText::new("Frequencies").strong());
                                        #[cfg(feature = "nvidia")]
                                        {
                                            ui.checkbox(&mut self.gpu_freq_graphics_vis, "GPU Graphics");
                                            ui.checkbox(&mut self.gpu_freq_sm_vis,       "GPU SM");
                                            ui.checkbox(&mut self.gpu_mem_effective,     "Show memory as effective (x2)");
                                            ui.checkbox(&mut self.gpu_freq_mem_vis,      "GPU Memory");
                                            ui.checkbox(&mut self.gpu_freq_video_vis,    "GPU Video");
                                        }
                                    });
                                });
                            });
                        } else {
                            egui::CollapsingHeader::new(g.display.clone()).id_source(format!("grp_other_{}", g.display)).default_open(false).show(ui, |ui| { for it in &mut g.items { ui.checkbox(&mut it.visible, &it.name); } });
                        }
                        if cols > 1 { ui.end_row(); }
                    }
                });
            });
        });
    }
}

impl App {
    fn footer_legend(&self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Legend:").strong());
            for g in &self.groups { if !g.visible { continue; } for it in &g.items { if !it.visible { continue; }
                let mut text = it.name.clone();
                let hot = self.temp_series[it.idx].last_y().map(|y| y >= g.hot).unwrap_or(false);
                let warn = self.temp_series[it.idx].last_y().map(|y| y >= g.warn).unwrap_or(false);
                if hot { text.push_str(" 🔥"); } else if warn { text.push_str(" 🥵"); }
                ui.horizontal(|ui| { ui.colored_label(it.color, "●"); ui.label(text); });
            }}
        });
    }
    fn side_legend(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("Legend").strong());
                for g in &self.groups { if !g.visible { continue; } for it in &g.items { if !it.visible { continue; }
                    let mut text = it.name.clone();
                    let hot = self.temp_series[it.idx].last_y().map(|y| y >= g.hot).unwrap_or(false);
                    let warn = self.temp_series[it.idx].last_y().map(|y| y >= g.warn).unwrap_or(false);
                    if hot { text.push_str(" 🔥"); } else if warn { text.push_str(" 🥵"); }
                    ui.horizontal(|ui| { ui.colored_label(it.color, "●"); ui.label(text); });
                }}
            });
        });
    }
}

// ===================== Entry =====================
fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 700.0])
            .with_min_inner_size([820.0, 560.0])
            .with_title("SIA - System Information Analyzer - By David Crawley - dc@ubiquityrobotics.com"),
        ..Default::default()
    };
    eframe::run_native("SIA - System Information Analyzer", options, Box::new(|_cc| Ok(Box::new(App::new(5 * 60, 1.0)))) )
}
