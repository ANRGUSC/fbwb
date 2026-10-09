"""Reconstruct flybrain inputs from the official v783 per-synapse source.

IDs in this particular export omit the fixed 720575940 prefix; reconstruct
them using int64 arithmetic. Weight normalization follows the pinned upstream
loader. All valid rows are retained, including parallel synapses.
"""
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DTYPE = np.dtype([('pre', '<i4'), ('post', '<i4'), ('size', '<f4')])


def restore_ids(series):
    return series.astype('int64') + np.int64(720575940000000000)


def main():
    raw = ROOT / 'data/raw'
    out = ROOT / 'data/prepared'
    out.mkdir(parents=True, exist_ok=True)
    neuron_source = pd.read_csv(raw / 'neurons.csv.gz', dtype={'root_id': 'int64'})
    types = pd.read_csv(raw / 'consolidated_cell_types.csv.gz', dtype={'root_id': 'int64'})
    neurons = neuron_source[['root_id', 'nt_type']].merge(types[['root_id', 'primary_type']], how='left', on='root_id', validate='one_to_one', sort=False)
    neurons.to_csv(out / 'neurons.csv', index=False)
    index = pd.Index(neurons.root_id)
    if not index.is_unique:
        raise ValueError('Duplicate neuron IDs')
    columns = ['pre_root_id_720575940', 'post_root_id_720575940', 'size']
    count = dropped = total = 0
    maximum = 0.0
    started = time.monotonic()
    # Disk spool bounds Python-object memory and supports a single gzip pass.
    with (out / 'synapses.bin').open('wb') as stream:
        for chunk in pd.read_csv(raw / 'fafb_v783_princeton_synapse_table.csv.gz', usecols=columns, dtype={columns[0]: 'int64', columns[1]: 'int64', 'size': 'float32'}, chunksize=1_000_000):
            pre = index.get_indexer(restore_ids(chunk[columns[0]]))
            post = index.get_indexer(restore_ids(chunk[columns[1]]))
            sizes = chunk['size'].to_numpy()
            if not np.isfinite(sizes).all() or (sizes < 0).any():
                raise ValueError('Invalid synapse size')
            valid = (pre >= 0) & (post >= 0)
            rows = np.empty(int(valid.sum()), dtype=DTYPE)
            rows['pre'], rows['post'], rows['size'] = pre[valid], post[valid], sizes[valid]
            rows.tofile(stream)
            count += len(rows)
            total += len(chunk)
            dropped += int((~valid).sum())
            if len(rows):
                maximum = max(maximum, float(rows['size'].max()))
            print(f'{total:,} source rows; {count:,} retained; {time.monotonic()-started:.1f}s', flush=True)
    if count == 0:
        raise ValueError('No synapses match neuron IDs')
    rows = np.memmap(out / 'synapses.bin', mode='r', dtype=DTYPE)
    weights = np.clip(rows['size'] / (maximum + 1e-6), 0.1, 2.0).astype(np.float32)
    inhibitory = neurons.nt_type.eq('GABA').to_numpy()
    # Block this operation to avoid an unnecessary full-size temporary array.
    for start in range(0, count, 1_000_000):
        stop = min(count, start + 1_000_000)
        weights[start:stop] *= np.where(inhibitory[rows['pre'][start:stop]], -1.0, 1.0)
    print('Coalescing parallel synapses into sparse connectivity...', flush=True)
    matrix = sp.coo_matrix((weights, (rows['pre'], rows['post'])), shape=(len(neurons), len(neurons))).tocsr()
    sp.save_npz(out / 'connectivity.npz', matrix)
    report = {'dataset': 'FAFB v783', 'neurons': len(neurons), 'source_synapses': total, 'retained_synapses': count, 'dropped_unknown_ids': dropped, 'directed_pairs': matrix.nnz, 'max_size': maximum, 'subsampling': False, 'weight_rule': 'clip(size / (max_size + 1e-6), 0.1, 2.0), negative for presynaptic GABA; sum duplicate pairs', 'neuron_order': 'official neurons.csv.gz row order; upstream private CSV ordering is unavailable', 'elapsed_seconds': time.monotonic()-started}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
