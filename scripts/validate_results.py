"""Assert flight-level acceptance criteria and summarize measured results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = {}
    for mode in ['hover', 'scripted', 'upstream', 'stable', 'corrected']:
        d = pd.read_csv(ROOT / 'results' / mode / 'telemetry.csv')
        assert np.isfinite(d.to_numpy()).all(), mode
        assert d.t.diff().dropna().between(0.019999, 0.020001).all(), mode
        assert d[['roll', 'pitch']].abs().max().max() < 0.2, mode
        assert abs(d.z.iloc[-1]-1) < 0.02, mode
        active = d[d.t >= 6]
        progress = d.x.iloc[-1]-d.x.iloc[0]
        reports[mode] = {'final_x_m': float(d.x.iloc[-1]), 'final_y_m': float(d.y.iloc[-1]), 'final_height_m': float(d.z.iloc[-1]), 'final_yaw_rad': float(d.yaw.iloc[-1]), 'max_tilt_rad': float(d[['roll','pitch']].abs().max().max()), 'max_spikes_per_step': float(d.spikes.max()), 'mean_active_brain_ms': float(active.brain_ms.mean())}
        if mode == 'hover':
            assert d[d.t >= 12].z.sub(1).abs().max() < 0.01
            assert abs(progress) < 0.01
        if mode == 'scripted':
            assert 0.7 < progress < 0.9
            assert 0.9 < d.yaw.iloc[-1] < 1.1
        if mode == 'upstream':
            assert d.spikes.max() > 0 and d.dn_activity.max() > 0
            assert 0.03 < progress < 0.4
            assert d.vx_target.abs().max() <= 0.3
            assert d.yaw_target.abs().max() <= 1
            error = d[d.t >= 7.5].vx.sub(d[d.t >= 7.5].vx_target).abs().max()
            reports[mode]['late_speed_tracking_error_mps'] = float(error)
            assert error < 0.08, error
        if mode == 'stable':
            assert d.spikes.max() == 0 and d.dn_activity.max() == 0
            assert abs(progress) < 0.01
        if mode == 'corrected':
            assert 0.01 < progress < 0.4
            assert d.dn_activity.max() > 0
            summary = json.loads((ROOT/'results/corrected/summary.json').read_text())
            assert summary['airborne_contact_steps'] == 0
            assert summary['scene'] == 'corridor'
            reports[mode]['forward_progress_m'] = float(progress)
    diagnostics = json.loads((ROOT/'results/diagnostics/full.json').read_text())
    assert diagnostics['upstream']['uniform']['max_spikes_per_step'] > 0
    assert diagnostics['stable']['uniform']['max_spikes_per_step'] == 0
    reports['interpretation'] = 'Integration and corrected timing verified. Corrected mode uses explicit voltage units and seeded sensory activation; original mode is unstable, integration-only mode is silent. Biological fidelity and navigation remain unvalidated.'
    (ROOT/'results/validation.json').write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
