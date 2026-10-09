"""Read-only probes for the official assets used by this integration."""
import gzip
import urllib.request

base = 'https://storage.googleapis.com/flywire-data/codex/data/fafb/783/'
for name in ['neurons.csv.gz', 'consolidated_cell_types.csv.gz', 'fafb_v783_princeton_synapse_table.csv.gz']:
    url = base + name
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r:
        print(name, r.status, r.headers.get('Content-Length'), flush=True)
    with urllib.request.urlopen(url, timeout=30) as r, gzip.GzipFile(fileobj=r) as stream:
        print(stream.readline().decode().strip(), flush=True)
        print(stream.readline().decode().strip(), flush=True)
