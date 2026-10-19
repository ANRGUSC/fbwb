# CPU FlyVis visual input and decoder comparison

The visual adapter is inspired by vib2810/fly-brain-zero-shot, pinned at
`0c63ba5cdd123f8d38eea62deb574353f0ec2632`. It uses the pretrained FlyVis
`flow/0000/000` model and that repository's derived eye/column maps and
per-cell-type moving-bar scales. It does not import their brain engine,
MuJoCo body, GPU/Warp implementation, or walking controller.

## Signal path

Six 96x96 Webots cameras form a cube at the drone's head position. Each is a
90-degree planar camera. The launcher creates an ignored `.flyvis-*.wbt` world
beside the original and adds these cameras to the robot's extension slot before
Webots loads it. Original worlds remain unchanged. Pixels are locally averaged and projected into
two sets of 721 angular samples. This preserves spatial brightness, without
computing optic flow outside the visual network. The front camera preview and
legacy flow diagnostics are still recorded; those features do not drive the
brain in FlyVis mode. During held/turn trials the new rig compensates yaw;
in free flight it rotates with the drone. The six cameras share one origin,
so this approximates angular eye coverage, not binocular parallax or full
compound-eye optics.

FlyVis runs entirely on CPU, in 10 ms integration steps within each 20 ms
Webots/brain period. The image is held over the two visual steps. The final
FlyVis activity drives the following brain window. Both visual and brain
states reset at the existing experiment phase boundaries. FlyVis resets to
a state warmed for one simulated second on luminance 0.8.

For each handoff neuron the external event rate is:

`150 Hz * clip(max(activity - background_activity, 0) / type_scale, 0, 1)`.

This is an engineered Poisson interface. Below-baseline responses are discarded;
it is not a validated conversion from membrane voltage to biological spikes.
The 27 handoff cell types include T4/T5, T2/T3, Tm and TmY neurons. The original
four-feature receptor stimulation is disabled when this handoff is active.

## What remains unchanged

`fbwb/corrected.py` is unchanged, including its membrane and synaptic equations,
weights, integration timestep, refractory rules, propagation delays, and
forward/climb decoding. `VisualFlyBrain` adds external input events and reads
the resulting spikes through a subclass. The original population decoder and
its existing symmetric optical-flow calibration remain intact. That calibration
is not refitted to FlyVis. Tests compare the legacy adapter with the original
brain and require exactly matching commands, voltages and spikes.

Unlike the reference repository, handoff cells do not receive special zero
refractory periods. They use this project's existing stimulation and refractory
semantics. The complete original graph remains connected, including recurrent
visual circuits; this is external drive into a complete graph, not replacement
of a surgically isolated optic-lobe subgraph.

## Mapping and provenance

The reference map estimates visual positions from Mi1 geometry and connectivity,
then stretches each cell type's directions across the eye field. These are
approximate registrations, not experimentally measured receptive fields.
Setup verifies source SHA-256 hashes, translates source row indices through
the pinned Shiu root-ID table, and joins to our local neuron table by exact
root ID. It never changes our connectome row order or annotations.

All 31,438 mapped IDs exist locally. Forty have conflicting cell-type labels;
setup excludes these from stimulation and lists them in
`data/flyvis_bridge/manifest.json`. The remaining 31,398 receive visual drive.
Missing IDs fail setup rather than silently shifting the mapping.

The optional environment pins FlyVis 1.1.3 and datamate 1.0.0. On Windows a
process-local HDF5 writer fixes datamate's attempt to unlink an open file.
No installed package or global settings are edited. Model downloads use the
FlyVis downloader's archive checksums. All downloaded assets stay ignored.

## Turn readouts

Both readouts observe the same spikes in each run. Only the selected readout
changes the yaw request. Forward speed, altitude stabilization, fixture rules,
command filtering, and the four propeller controls remain as before.

* Population: existing gain-balanced left/right motion-population mean rates,
  measured over the 20 ms neural window.
* DNa02: left minus right DNa02 rates, averaged over a trailing 50 ms window
  (zero padded immediately after reset). The proposal is
  `clip(gain * (left_hz - right_hz) / 100, -0.8, 0.8)` rad/s, CCW positive.
  The default gain is 2, configurable with `--dna02-gain`. The reference's
  dimensionless walking asymmetry is adapted to yaw rate; this gain/sign must
  be assessed for the drone and is not a biological flight calibration.

The DNa02 window is updated at each unchanged 1 ms neural step. If either side
is absent, DNa02 cannot be selected and the comparison marks it unavailable.
The population decoder may be biased under the new input distribution; recording
both outputs exposes that rather than silently recalibrating the old decoder.

`scripts/compare_decoders.py RUN_DIRECTORY` writes a per-phase JSON report with
mean proposals, DNa02 rates, direction agreement when both are active, absolute
differences, and CPU timings. This is a same-spike comparison. Separate runs
with different applied decoders measure closed-loop outcomes, not identical-input
decoder differences after the trajectories diverge.

## Validation

Unit tests cover legacy equivalence, decoder-independent neural trajectories,
DNa02 sign/window/reset, ID remapping failures, camera-coordinate orientation,
rate rectification/saturation, and prevention of simultaneous receptor/handoff
drive. Integration runs should verify that real Webots images produce nonzero
handoff inputs, both decoder traces are finite, the selected decoder reaches
the existing filter/PID, and the fixture releases normally at 12 s.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/validate_visual_run.py results/flyvis-dna02
.\.venv\Scripts\python.exe scripts/compare_decoders.py results/flyvis-dna02
```

The launcher automatically writes `decoder_comparison.json` and a PNG plot
after a completed corrected-mode run with neural samples.

Neither the historical optical-flow acceptance numbers nor the reference's
walking results establish navigation performance for this new visual input.

### Initial CPU/Webots observation (2026-10-18)

A 12.4-second explained-scene run with DNa02 selected completed all visual
phases and 0.4 seconds after free-flight release. It passed the signal-path
checks over 321 neural frames, with zero airborne contacts and final altitude
0.984 m. Controller runtime was 331 seconds, excluding model startup.
Mean active brain computation was 913 ms per 20 ms frame; FlyVis typically
added 63–75 ms in these phases.

The two raw proposals were substantially different on the same spikes:

| Stimulus phase | Population mean yaw (rad/s) | DNa02 mean yaw (rad/s) |
| --- | ---: | ---: |
| Left-screen yaw trial | +0.365 | -0.396 |
| Right-screen yaw trial | +0.324 | -0.012 |
| First 0.4 s of free flight | +0.359 | -0.381 |

Positive means left/CCW. The population readout remained left-biased; DNa02
was right-biased or approximately silent. DNa02 cells settled near 240–260 Hz
in much of this run, including static-scene trials. Therefore the integration
works, but this is **not successful direction-selective steering** or evidence
that DNa02 is a better controller in our unchanged brain. The high persistent
rates and the old population calibration are visible limitations to investigate
in a separate experiment. No brain parameters were changed to improve this
result. Full recordings remain local/ignored and can be reproduced with the
commands above (`--duration 12.4` for this abbreviated run).

Additional corridor checks completed 6.4 simulated seconds each with
FlyVis/population and optic-flow/population, including 21 neural frames of
free flight. Both passed signal-path validation and recorded zero airborne
contacts. All 26 unit tests passed. These are short integration checks, not
long-horizon navigation trials.
