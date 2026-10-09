# Flybrain in Webots

A working Webots interface for `dylankainth/flybrain`, branch
`evolutionary-behavior`, pinned to **97e02fb7c8d19fd5267c3fa9194f8001e08fb613**.
Uses the official Webots R2025a Crazyflie and Bitcraze's velocity/altitude PID.

The default launcher uses **corrected, unit-consistent neural dynamics** and a
structured corridor. The original upstream model and an integration-only
diagnostic remain available for comparison. Timing/numerical corrections are
tested separately from biological validity and navigation performance.

## Run on this machine

- Double-click **Run Hover.cmd** for a 30-second hover demonstration.
- Double-click **Run Flybrain.cmd** to run corrected neural dynamics in the corridor.
- Double-click **Run Explained Demo.cmd** for visual probes, neural left/right turns, then free flight.
- Double-click **Run Stable Diagnostic.cmd** to compare stable integration.
- **Run Upstream Comparison.cmd** reproduces the original unstable model.

Each run opens Webots, performs a finite experiment, saves results, and closes
that simulator instance. Brain runs spend the first six simulation seconds
taking off under the ordinary PID, then release forward/yaw control to the
brain. The neural state starts at rest at handoff. The full graph is CPU-bound;
simulation slows down while preserving neural and physical time alignment.
There is no artificial forward command in either brain mode.

```powershell
.\.venv\Scripts\python.exe scripts/run.py --mode hover --duration 30
.\.venv\Scripts\python.exe scripts/run.py --mode scripted --duration 15
.\.venv\Scripts\python.exe scripts/run.py --mode upstream --duration 8
.\.venv\Scripts\python.exe scripts/run.py --mode stable --duration 8
.\.venv\Scripts\python.exe scripts/run.py --mode corrected --duration 8
```

Add `--minimize` for batch runs or `--output results/my-run` to keep a separate
experiment. Default output directories are reused per mode. Run one simulator
at a time. `--scene corridor` is the default. `--scene diagnostic` selects the
original textured test enclosure. Opening either world directly defaults to hover.

Webots: `C:\Program Files\Webots\msys64\mingw64\bin\webots.exe`.
The project has its own `runtime.ini`, pointing to `.venv\Scripts\python.exe`;
global Webots preferences are not changed.

## Connection

```text
camera BGRA -> grayscale + Farneback flow -> 4 upstream sensory features
  -> selected brain.compute() -> forward/yaw RC outputs
  -> bounded body velocity targets -> Bitcraze PID -> four propellers
                                      ^
                                IMU + gyro + GPS
```

- Physics/controller period: 20 ms; 20 neural substeps of 1 ms per brain call.
- Camera: stock Crazyflie camera, resampled to 160 x 120 for optic flow.
- Upstream features: forward radial flow, left/right horizontal flow ratios,
  vertical flow ratio. A **0.01 pixel/frame noise floor** is added before ratio
  normalization; otherwise identical images can produce saturated features.
- Forward output maps `[-100,100]` to `[-0.3,0.3]` m/s.
- Tello clockwise-positive yaw maps to Webots counterclockwise-positive yaw
  with a sign inversion, capped at 1 rad/s.
- Lateral velocity is zero; altitude ramps to 1 m and stays under PID control.
- Neural velocity/yaw targets pass through a 0.2 s low-pass filter and slew
  limits of 0.3 m/s² and 1 rad/s² before the PID. This is an adapter change:
  raw neural-window jitter otherwise excited the PID derivative terms and
  produced substantial speed overshoot. Both raw and filtered targets are logged.
- The `vertical` output is displayed in the panel and recorded in `live.ndjson`; it
  does not drive height. The motor sign pattern is `[-,+,-,+]`.
- The corridor has broad blue visual panels, an offset orange obstacle, a
  1.4 m doorway, and a green destination marker. Floor tiles provide optic flow
  without random pixel noise. No ROS, PX4, Tello networking, or
  separate server is needed. No obstacle avoidance is added by the adapter.

The marker is a visual landmark, not an instruction encoded in the brain.
Forward progress, airborne contacts, and doorway passage are measured by the
simulator but are not supplied as rewards or steering commands. A short run
verifies the connection, not successful navigation through the doorway.

## Data and upstream reproducibility

`vendor/flybrain` is an unmodified checkout. `upstream.json` pins the source
commit and SHA-256. Only its `RealFlyBrain` class is loaded through Python AST;
the Tello script's module-level loading, networking, UI, and flight entry points
are never executed. Model source drift is rejected.

The repository documents `download_data.py` and `parse_flywire.py`, but neither
is present at this commit. This project reconstructs the inputs from public
FAFB v783 files hosted by FlyWire:

- `neurons.csv.gz`
- `consolidated_cell_types.csv.gz`
- `classification.csv.gz` (anatomical soma-side annotations for corrected mode)
- `fafb_v783_princeton_synapse_table.csv.gz`

Download URLs and hashes are saved in `data/raw/manifest.json`. The prepared
manifest records **139,255 neurons, 80,215,790 synapse rows, 19,912,747 directed
pairs, zero dropped rows, and no subsampling**. Source neuron order is retained.
The ID suffixes in the synapse export are reconstructed with integer arithmetic.
Synapses use the upstream rule `clip(size/(max_size+1e-6), 0.1, 2.0)` with a
negative sign for presynaptic GABA, then duplicate pairs are summed.

The author's private processed CSVs are unavailable. In particular, their row
order cannot be reproduced. Since upstream divides receptor and motion-neuron
lists by index rather than anatomical side, changing row order can change yaw.
This is a faithful implementation of the published code/weight rule on the
official data, **not a bit-for-bit reproduction of the author's private inputs**.

Data stays local and is excluded from source control. See the
[FlyWire data source](https://codex.flywire.ai/api/download) and its linked
attribution/usage terms before redistributing data.

## Numerical findings

### Corrected mode (default launcher)

`fbwb/corrected.py` implements the voltage-based LIF equations from
[Shiu et al.'s reference model](https://github.com/philshiu/Drosophila_brain_model/blob/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/model.py):

```text
dv/dt = (V_rest - v + g) / 20 ms
dg/dt = -g / 5 ms
synaptic event: g += signed_synapse_count * 0.275 mV
```

The coupled linear equations are integrated analytically between events, with
no voltage clamp masking divergence. Rest/reset is -52 mV and threshold is
-45 mV. Refractory neurons hold their voltage; synaptic state is reset on spikes.
At the default 1 ms step, the reference 1.8 ms delay is quantized to 2 ms and
2.2 ms refractory duration to 3 ms. This is explicit discretization, not exact
reproduction of the reference simulator's event scheduling. The 20 ms flight
period always advances 20 ms of neural time, regardless of wall-clock speed.

The old current gains were dimensionally incompatible. Corrected mode instead
maps the same four visual features to seeded Poisson receptor activation up to
150 Hz, using the reference's strong external activation interface. No tonic
forward drive is added. The reference uses synapse counts, so `prepare_counts.py`
builds count-based weights from all 80,215,790 synapses. Original size-normalized
weights are preserved for the two comparison modes. The GABA-only negative sign
convention remains a limitation. Corrected mode now uses anatomical side labels
for receptor drive and motion-cell readout, as described below.

### Bidirectional turn decoder

The original index-based readout favored left turns in the initial demo.
Corrected mode joins `classification.csv.gz` by exact root ID, preserving the
matrix order. R1-6 input groups and motion pools use annotated left/right sides.
These labels describe soma side, not a reconstructed visual-field map; see
[FlyWire's annotation definitions](https://codex.flywire.ai/faq).

Mean motion-cell rates remove pool-size bias. Because the two hemispheres have
different response gains in this model, `scripts/calibrate_turn.py` measures
each side's response to the SAME symmetric input (200 ms warmup, 400 ms fit).
Current reference rates are 17.466 and 35.031 Hz/cell. The decoder uses:

```text
L = mean left motion-cell Hz / left reference Hz
R = mean right motion-cell Hz / right reference Hz
Tello yaw = 100 * tanh(2 * (R-L) / max(R+L, 1))
Webots raw yaw target = -Tello yaw / 100 rad/s
```

Only measured neural spikes enter this readout. No phase name, camera feature
or forced sign overrides yaw. Independent unilateral validation gives about
-54 RC units for left stimulus and +52 for right; symmetric input stays within
3 RC units and zero input is silent, across seeds 7 and 19. Unilateral inputs
are not used to fit the references. Calibration records source hashes and is
rejected if stale or unvalidated. This is an engineering decoder, not a claim
that anatomical motion pools implement the fly's biological steering circuit.

Tests verify exact passive relaxation and synaptic decay at 1, 0.5, and 0.2 ms,
event delay/sign, refractory behavior, seeded repeatability, zero-input silence,
and matching of control and neural time. Full-graph tests show zero-input silence
and stimulus-driven receptor-to-descending activity. Activity can persist after
the stimulus ends. Strong inhibition also produces unrealistically negative
voltages in some neurons (around -1 V in the diagnostic). These are unresolved
network/input-model limitations; a numerical fix does **not** establish
physiologically valid behavior or goal-directed navigation. We retain these
results rather than conceal them with voltage clipping.

### Original mode and integration-only diagnostic

The upstream parameters imply `tau_m = R_m*C_m = 10*2e-6 = 20 microseconds`,
while the explicit Euler step is 1 millisecond. The passive update's error
multiplier is **-49**, so tiny perturbations grow and alternate sign until
voltage clipping or spike resets intervene. The isolated-neuron test confirms
this amplification. Finite, clipped state does not establish numerical validity.

`StableFlyBrain` uses an exponential membrane update with input held constant
over each step. It retains upstream connectivity, sensory currents, synaptic
update, thresholds, reset, refractory logic, and readout. It is a numerical
diagnostic, not a physiological correction. Maximum upstream photoreceptor
current produces only nanovolts of equilibrium depolarization with `R_m=10`,
far below the 7 mV threshold gap; simply stabilizing integration should not be
expected to create useful activity. Fixing resistance/capacitance/current units
and synaptic scaling requires a separate model-validation decision.

CPU inference uses the same sparse weights in CSR layout. The recorded
benchmark on this machine measured about 0.422 s for COO versus 0.0374 s for
CSR per matrix-vector product, with zero maximum difference on the test vector.
Other inputs/platforms may differ at floating-point roundoff level.

## Verification and artifacts

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/diagnose_brain.py
.\.venv\Scripts\python.exe scripts/diagnose_brain.py --full
.\.venv\Scripts\python.exe scripts/diagnose_corrected.py
.\.venv\Scripts\python.exe scripts/benchmark_sparse.py
.\.venv\Scripts\python.exe scripts/validate_results.py
```

Tests cover passive stability/instability, neural time alignment, command axes
and limits, stationary/moving camera images, and exact 64-bit ID reconstruction.
`results/diagnostics/isolated.json` and `full.json` compare zero, uniform, left,
and right stimuli. The isolated graph is explicitly synthetic and is only used
for numerical diagnostics, never as a substitute for the flight connectome.

Each simulation produces `summary.json`, `telemetry.csv`, `camera.png`, and
`scene.png` under `results/<mode>`. Summaries carry a unique run ID so stale
results cannot be mistaken for a successful new run. The full telemetry records
position, orientation, speed, commanded speed/yaw, visual features, neural
activity, and inference time. The camera/scene snapshots are taken at handoff.

Verified baseline: the 15-second scripted test moved forward approximately
0.798 m, executed the yaw command, and ended at 0.99944 m altitude, with peak
roll/pitch below 0.042 rad. The 30-second hover test settled at 1 m. See generated
result files for the full-brain experiment results and measured runtime.

The original-mode run with command smoothing achieved 0.156 m forward progress
in two seconds after handoff, with late speed-tracking error under 0.0015 m/s.
This validates the interface, not its unstable neural model. Corrected-mode
results are recorded separately in `results/corrected` and
`results/diagnostics/corrected.json`.

Corrected corridor run: 8 s simulated (6 s takeoff + 2 s neural control),
0.0752 m forward progress, final altitude 0.99868 m, peak roll/pitch 0.0208 rad,
and zero airborne contacts. It did not reach the doorway. This took about 93 s
wall time on the CPU; active brain calls averaged about 881 ms per 20 ms of
simulation. The scene is visible during runs, but this machine does not run the
full connectome in real time. `results/comparison.png` summarizes the three
integration checks: the original modes used the test room, while corrected mode
used the corridor. It is not a controlled comparison of navigation performance.
The original thirteen focused tests and five-mode flight checks pass.

## Controlled trials and simultaneous visualization

Run `Run Explained Demo.cmd` for the controlled trials followed by free flight.
Webots is placed on the left and a native desktop measurement panel on the right.
The eighteen-second simulated experiment takes several minutes on this CPU. See
[WATCH_GUIDE.txt](../WATCH_GUIDE.txt) for the phase-by-phase viewing narrative.

The panel displays the camera, four measured optical-flow inputs, 144 sampled
receptor neurons with IDs and measured firing rates, full-population activity,
spike counts, four decoded output axes, applied flight targets and rotor speeds.
It distinguishes requested receptor drive from observed firing, and neural
outputs from altitude PID, fixture and command-filter effects. These are sampled
cells with anatomical side labels, not reconstructed anatomical ommatidia.
The source's ratio features can cross-activate even in controlled visual trials.

The new `worlds/explain.wbt` contains two broad coloured motion screens. At 6 s,
the supervisor holds the drone near a fixed pose for static, left, right,
outward and vertical trials. Each held trial resets brain state and image
history. Rotors and the altitude PID continue running to preserve the flight
handoff. At 8-10 s and 10-12 s, left/right stimuli drive actual neural yaw on a
rig that holds position and releases yaw. A stabilized test camera isolates the
visual input from the rig's rotation. Each trial starts with centred pose and
reset brain state. At 12 s, the centred drone is released, brain state is reset,
and its onboard camera drives forward/yaw in free flight until 18 s.
Neural climb remains unused,
sideways remains disabled and the inherited forward decoder cannot reverse.

`live.ndjson` records synchronized camera/brain/command frames, `latest.json`
publishes the latest frame atomically, and `narrative.txt` records explanations
every 0.2 simulated seconds and at phase changes. The panel supports replay,
scrubbing, jumping directly to a named phase, pausing its view and stopping the run. `Replay Explained Demo.cmd`
reopens the last recording. Webots closes at the finite duration; the panel
stays open. These measurements explain the implemented signal path, not a
validated biological navigation strategy or a complete causal neural account.

Current recordings are in `results/explained-bilateral`; the earlier short demo
is retained in `results/explained-demo`. A shareable explanation of all adapter
and neural-model changes is in [STUDENT_SUMMARY.txt](../STUDENT_SUMMARY.txt).

Validation: `python -m pytest -q` runs 19 focused checks;
`python scripts/validate_explained.py` checks the full recorded experiment,
including stimulus separation, held commands, actual spike totals, 20 ms
alignment, measured left AND right rotation, and six seconds of free flight.
Rebuild the visual-trial world using
`python scripts/build_explain_world.py`.

Verified bilateral demo: 19 tests and full-recording acceptance checks pass.
The drone turns +48.6 degrees in LEFT TURN and -47.1 degrees in RIGHT TURN,
then flies 0.294 m over six simulated seconds using the onboard camera, with
zero airborne contacts and final altitude 0.995 m. The recording contains 660
published frames. See `results/explained-bilateral/acceptance.json`; the panel
phase selector can jump directly to each segment.

Historical initial demo (before anatomical turn decoding): 17 tests and the recorded-run acceptance checks passed.
The 260 published frames include a silent static baseline, left/right feature
separation above 0.99 versus below 0.011, and a collision-free release. During
free flight the drone advances 0.078 m in world X; final altitude is 0.946 m,
maximum altitude error after takeoff is 0.066 m and peak roll/pitch is 0.0193 rad.
The controller loop took 167 s wall time for 10 s simulated. These short-trial
checks establish integration behavior, not long-duration physiological stability.

## Rebuild

Python 3.12 is used here. This machine's environment reuses its installed CPU
PyTorch through `venv --system-site-packages`; OpenCV is installed in the venv.
For a standalone installation, create a regular venv and install requirements.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/fetch_assets.py --flybrain
.\.venv\Scripts\python.exe scripts/fetch_assets.py --data
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/prepare_counts.py
.\.venv\Scripts\python.exe scripts/calibrate_turn.py
.\.venv\Scripts\python.exe scripts/build_world.py
.\.venv\Scripts\python.exe scripts/build_explain_world.py
```

The source hash in `upstream.json` normalizes UTF-8 source to LF line endings,
so platform line-ending conversion does not require changing the pin.
The downloader needs about 2.7 GB, preparation stores a roughly 1 GB
binary spool plus the sparse graph. `prepare_data.py` replaces its generated
outputs if rerun. Do not run preparation and inference simultaneously.

## Attribution

- [dylankainth/flybrain](https://github.com/dylankainth/flybrain/tree/evolutionary-behavior): neural class, sensory features, and readout (README declares MIT).
- [Cyberbotics Webots R2025a](https://github.com/cyberbotics/webots/tree/R2025a/projects/robots/bitcraze/crazyflie): Crazyflie assets, licensed for use with Webots; original headers retained.
- Bitcraze AB: `pid_controller.py`, MIT, original header retained.
- FlyWire Consortium: FAFB v783 connectivity, neurotransmitter predictions, and cell annotations. See upstream/data-source citation guidance.
- [Shiu et al. model repository](https://github.com/philshiu/Drosophila_brain_model): a reference for subsequent validation; this integration does not claim to reproduce its published model.
