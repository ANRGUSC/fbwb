# Flybrain + Webots (fbwb)

Connect the [Flybrain evolutionary-behavior model](https://github.com/dylankainth/flybrain/tree/evolutionary-behavior)
to a simulated Crazyflie drone in Webots. Includes corrected neural dynamics,
an anatomical, gain-balanced turn decoder, structured visual trials, and a
desktop panel for live receptor activity, neural spikes, motor outputs and replay.

Maintained by [bkrishnamachari](https://github.com/bkrishnamachari) at ANRG, USC.

**This repository contains our integration code, worlds, tests and documentation.**
It does not contain Flybrain's source tree, Webots itself, downloaded robot/PID
assets, FlyWire datasets, Python environments or simulation recordings. Setup
downloads these dependencies separately into ignored local directories.

## Install and connect the dependencies

The launcher, `.cmd` shortcuts, window docking and desktop panel currently target
**Windows 10/11 with Python 3.12**. Other platforms are not yet supported by the
launcher, although much of the neural code is platform independent.

Install:

- [Git](https://git-scm.com/downloads) and [Python 3.12](https://www.python.org/downloads/).
  Include Python's Tcl/Tk support for the panel.
- [Webots R2025a](https://github.com/cyberbotics/webots/releases/tag/R2025a), using
  its Windows installer. The default location is `C:\Program Files\Webots`.

Run the following in PowerShell. No virtual-environment activation is required:

```powershell
git clone https://github.com/anrgusc/fbwb.git
cd fbwb
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/fetch_assets.py --flybrain
.\.venv\Scripts\python.exe -m pytest -q
```

`fetch_assets.py --flybrain`:

1. Clones Flybrain into `vendor/flybrain` and checks out commit
   `97e02fb7c8d19fd5267c3fa9194f8001e08fb613`. Its source checksum is verified
   after normalizing line endings. It does not run the original Tello program
   or install its hardware/networking dependencies.
2. Downloads only the Crazyflie model, meshes, propeller texture and Bitcraze PID
   from the Webots `R2025a` source tag. It stores these in `protos/` and
   `controllers/flybrain_webots/pid_controller.py`; these files stay untracked.
3. Rewrites one robot texture URL to use the downloaded local texture. No global
   Webots configuration is modified.

If Webots is installed elsewhere, set its installation directory before launching:

```powershell
$env:WEBOTS_HOME = 'D:\Applications\Webots'
```

Check the simulator connection without downloading the connectome:

```powershell
.\.venv\Scripts\python.exe scripts/run.py --mode hover --duration 8
```

The launcher writes a local, ignored `runtime.ini` so Webots uses this project's
Python interpreter. The supplied worlds load the downloaded Crazyflie PROTO;
our controller reads its camera/IMU/GPS/gyro and drives its four motors through
the downloaded PID controller.

## Download and prepare the full connectome

The full neural demo requires a multi-GB download and several GB of RAM/disk.
It uses all 139,255 prepared neurons; there is no small substitute brain.

```powershell
.\.venv\Scripts\python.exe scripts/fetch_assets.py --data
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/prepare_counts.py
.\.venv\Scripts\python.exe scripts/calibrate_turn.py
```

These commands download official FlyWire FAFB v783 neuron, cell-type,
classification and per-synapse exports, reconstruct the sparse graph, prepare
signed synapse-count weights and calibrate the two anatomical motion-pool gains
with symmetric stimulation. Calibration also validates held-out left/right,
symmetric and zero inputs using two random seeds.

The original Flybrain author's private processed data is not included in their
repository. We reconstruct inputs from public exports; this is not a bit-for-bit
reproduction of that private dataset. URLs and hashes are recorded locally.
If public download access changes, obtain the same named v783 files through
[FlyWire Codex](https://codex.flywire.ai/api/download) and place them in `data/raw/`.
See [third-party sources](THIRD_PARTY.md) for attribution and upstream terms.

## Run the demo

The default corrected-mode visual input is now **CPU FlyVis**. The original
optic-flow/receptor input remains available as `--vision optic-flow`. The LIF
brain equations, connectome weights, timing, forward decoder, and altitude PID
are unchanged. Both turn decoders are measured on the same spikes in every
corrected-mode run; `--decoder population` (default) or `--decoder dna02`
selects which one actually drives yaw.

Install the optional visual dependencies and prepare the pretrained model/maps:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-flyvis.txt
.\.venv\Scripts\python.exe scripts/setup_flyvis.py --download-models
```

Use **Run FlyVis Population.cmd**, **Run FlyVis DNa02.cmd**, or
**Run Legacy Optical Flow.cmd**. **Run Explained Demo.cmd** now runs FlyVis
with the population decoder. The CPU front end adds startup and per-frame work;
simulation time remains synchronized with the unchanged brain.

```powershell
# New vision; original turn readout (both readouts are logged)
.\.venv\Scripts\python.exe scripts/run.py --mode corrected --vision flyvis --decoder population --scene explain --duration 18 --panel --output results/flyvis-population
# New vision; DNa02 controls yaw
.\.venv\Scripts\python.exe scripts/run.py --mode corrected --vision flyvis --decoder dna02 --scene explain --duration 18 --panel --output results/flyvis-dna02
# Original input and output behavior
.\.venv\Scripts\python.exe scripts/run.py --mode corrected --vision optic-flow --decoder population --scene explain --duration 18 --panel --output results/explained-bilateral
# Compare raw proposals on the SAME neural trajectory
.\.venv\Scripts\python.exe scripts/compare_decoders.py results/flyvis-population
```

`decoder_comparison.csv` contains both raw yaw proposals in rad/s (+ = left),
DNa02 left/right rates, selected decoder, actual applied yaw, and CPU timing.
The panel shows the same comparison and the externally driven visual neurons.
These paired proposals isolate decoder differences. Comparing two independently
controlled flights also changes subsequent images and neural activity.
See [the FlyVis integration notes](docs/FLYVIS.md) for mapping assumptions and
the explicit graded-activity-to-spiking interface.

The following describes the original explained experiment. Its historical
acceptance numbers and `validate_explained.py` apply to the **optic-flow** path;
they are not acceptance criteria or promised outcomes for FlyVis/DNa02.

Double-click **Run Explained Demo.cmd**, or run:

```powershell
.\.venv\Scripts\python.exe scripts/run.py --mode corrected --vision optic-flow --scene explain --duration 18 --panel --output results/explained-bilateral
```

| Simulated time | What happens |
| --- | --- |
| 0–6 s | PID-controlled takeoff to one metre |
| 6–8 s | Held-pose static, left, right, outward and vertical visual probes |
| 8–10 s | Neural left turn on a yaw-only rig |
| 10–12 s | Neural right turn on the rig |
| 12–18 s | Fixture released; onboard-camera neural forward/yaw free flight |

The turn rig uses a stabilized test camera to isolate the stimulus from the
drone's rotation. Both turns are decoded from actual network activity, not
scripted motor commands. Free flight uses the ordinary onboard camera.

Webots and the panel appear side by side. The panel shows image features,
144 sampled receptor cells with IDs/rates, population activity, spike counts,
decoded versus applied outputs, actual heading, and a running explanation.
The full graph is CPU-bound: our 18-second test took about **nine minutes** of
wall time. Neural and physical time remain aligned while it runs slowly.

The finite simulation closes Webots when finished; the panel remains open.
Click **Replay** and select **LEFT TURN**, **RIGHT TURN** or **FREE FLIGHT** in
the phase dropdown. **Replay Explained Demo.cmd** reopens the last recording.
Pause view only pauses the display; Stop run stops the simulator after its
current neural computation. [WATCH_GUIDE.txt](WATCH_GUIDE.txt) explains what to watch.

Other launchers provide hover, a corridor flight, original upstream dynamics and
a stability-only diagnostic. `--output results/my-run` preserves a separate
recording; otherwise output folders are reused. Run one simulator at a time.

## Changes, validation and limitations

[STUDENT_SUMMARY.txt](STUDENT_SUMMARY.txt) is a brief explanation of every major
change and why it was needed. [docs/TECHNICAL.md](docs/TECHNICAL.md) contains the
equations, timing, provenance, decoder formula and historical experiments.

The verified bilateral run passed 19 unit tests and recorded-flight acceptance
checks: **48.6° left, 47.1° right**, then **0.294 m over six simulated seconds**
of free flight, zero airborne collisions and final altitude **0.995 m**. These
are measurements from the development machine, not guaranteed results on all
hardware/dependency versions. Large recordings are not committed; reproduce them:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# Run the explained demo above, then:
.\.venv\Scripts\python.exe scripts/validate_explained.py
```

Logs include `telemetry.csv`, `live.ndjson`, `narrative.txt`, `summary.json` and
`acceptance.json`. The last file is created by the acceptance checker.

This is an experimental engineering interface, **not a validated biological
steering model**. Side annotations are soma-side labels, not an ommatidial visual
field map. The four coarse optical-flow features can cross-activate. The model
retains GABA-only inhibition, persistent activity and unrealistically negative
voltages in some cells. Altitude uses the PID; neural climb is displayed but
unused, sideways is disabled and the forward decoder cannot reverse. No learned
goal policy or obstacle-avoidance system is added.

## Repository layout

| Path | Purpose |
| --- | --- |
| `fbwb/` | Neural models, sensory processing, decoding, experiments and telemetry |
| `controllers/flybrain_webots/` | Our Webots controller |
| `worlds/` | Our diagnostic/corridor/test worlds and generated textures |
| `scripts/` | Dependency downloads, preparation, calibration, launching, panel and validation |
| `tests/` | Small tests; no full dataset or simulator required |
| `upstream.json` | Pinned Flybrain commit and normalized source checksum |
| `vendor/`, `protos/`, `data/`, `results/`, `.venv/` | Local downloads/generated state, excluded from Git |

The three small texture PNGs are generated by our world builders, not copied
Webots assets. To regenerate our diagnostic/explanatory worlds and textures:

```powershell
.\.venv\Scripts\python.exe scripts/build_world.py
.\.venv\Scripts\python.exe scripts/build_explain_world.py
```

For reference, the tested environment used Python 3.12, NumPy 2.4.3, pandas
3.0.1, SciPy 1.17.1, CPU PyTorch 2.13.0, OpenCV 5.0.0.93, pytest 9.0.2 and
Matplotlib 3.10.8. `requirements.txt` supplies lower bounds rather than freezing
that machine's entire environment.
