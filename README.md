This software was born of annoyance. I couldn't find a utility that cleanly displayed system temperatures at the same time as telling me the utilization rates of system components. The graphical diagnostic systems that were available deluged their graphs with lots of useless information from multiple redundant places, and overall the things that I saw were inelegant, annoying to use and didn't really tell me the information I wanted. The first version was coded in a few hours.

---

# SIA - The System Information Analyzer

A clean, elegant, and useful system monitoring tool.

## What It Does

* Displays real-time system temperatures for CPU, GPU, and other thermal sensors.
* Shows utilization rates of key system components (CPU, GPU, memory, disk, etc.).
* Provides clean, aligned textual output — optimized for quick scanning in a terminal window.
* Designed to be hackable and extendable — add your own metrics, formats, or output modes easily.

---

## Why You’ll Like It

* Instant clarity: See what’s hot and what’s busy in one glance.
* Elegant simplicity: No graphs. No mouse clicks. Just signal, no noise.
* Developer-friendly: Written in Rust — small, fast, and easy to contribute to.
* Scriptable: Perfect for embedding in logs, scripts, or system dashboards.Contributing

I built this because I wanted something better — and I’d love your help making it even better. Pull requests, ideas, and performance tweaks are all welcome. If you think other tools are too noisy or annoying, you’re in the right place.

## Changes I'd like to see

* Pefromance improvements, performance improvements, performance improvements - a system monitoring tool should consume the minimum ofsystem resources
* Headless operation - provide an equivalent system monitoring features from a web-page that can be acceessed remotely
* CSV file triggering - so that data can be analyzed later
* Triggering inside the code - so you can correlate what your code is doing to how the system is performing

---

## License

BSD - 3 Clause.
