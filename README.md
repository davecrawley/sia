This software was born of annoyance. I couldn't find a utility that cleanly displayed system temperatures at the same time as telling me the utilization rates of system components. The graphical diagnostic systems that were available deluged their graphs with lots of useless information from multiple redundant places, used hard to decipher names for system resources and overall the things that I saw were inelegant, annoying to use and didn't really tell me the information I wanted. I quickly pulled together a verison that didn't have these shortcomings.

---

# SIA - The System Information Analyzer

A clean, elegant, and useful system monitoring tool.

## What It Does

* Displays real-time system temperatures for CPU, GPU, and other thermal sensors.
* Shows utilization rates of key system components (CPU, GPU, memory, etc.).
* Shows Current operting frequency rates across system components.

---

## Why You’ll Like It

* Instant clarity: See what’s hot and what’s busy in one glance.
* Elegant simplicity: Only display the information you need
* Developer-friendly: Written in Rust — small, fast, and easy to contribute to.

I built this because I wanted something better — and I’d love your help making it even better. Pull requests, ideas, and performance tweaks are all welcome. If you think other tools are too noisy or annoying, this is for you.

---

## Distribution

A .deb is available here:

https://github.com/davecrawley/sia/releases/download/Initial/sia_0.0.1-1_amd64.deb

## Quick install (Debian/Ubuntu)

```bash
wget https://github.com/davecrawley/sia/releases/download/Initial/sia_0.0.1-1_amd64.deb
sudo apt install ./sia_0.0.1-1_amd64.deb
```

After installation, run:

```bash
sia
```
Or just look for a picture of a singer with a bow on her head in your app finder.

## Build from source (optional)

```bash
git clone https://github.com/davecrawley/sia.git
cd sia
cargo build --release
```

The compiled binary will be in `target/release/sia`.

---

## How you can help

The number one thing we need help with is testing. It was designed and tested on a system that was intel/nvidia based but because the probes and outputs are so system dependent, we can't be sure whether they will still work on hardware from companies that do things a little differently. Even things like the UI could break on systems that have a different number of sensors than the one on which it was designed. So please try it and tell us whether its working right.
If you spot a bug or have an idea to make SIA better, open an [issue on GitHub](https://github.com/davecrawley/sia/issues) or submit a pull request.

Let’s keep system monitoring simple, elegant, and actually useful.

## Changes I'd like to see

* Pefromance improvements, performance improvements, performance improvements - a system monitoring tool should consume the minimum of system resources
* Headless operation - provide an equivalent system monitoring features from a web-page that can be acceessed remotely
* CSV file triggering - so that data can be analyzed later
* Triggering inside the code - so you can correlate what your code is doing to how the system is performing

---

## License

BSD - 3 Clause.
