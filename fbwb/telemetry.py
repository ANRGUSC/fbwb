"""Append-only recording and atomic latest-frame publication for the desktop UI."""
import base64
import json
from pathlib import Path
import time
import cv2


class Publisher:
    def __init__(self, directory, run_id):
        self.directory = Path(directory)
        self.run_id = run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory/'live.ndjson').open('w',encoding='utf-8')
        self.narrative = (self.directory/'narrative.txt').open('w',encoding='utf-8')
        self.last_phase = None
        self.last_note = -1.0
        self.seq = 0

    def publish(self, payload, frame):
        self.seq += 1
        packet = dict(payload, seq=self.seq, run_id=self.run_id, published_wall=time.time())
        _, encoded = cv2.imencode('.png',cv2.resize(frame[:,:,:3],(144,144)))
        packet['camera_png'] = base64.b64encode(encoded).decode('ascii')
        text = json.dumps(packet,allow_nan=False,separators=(',',':'))
        self.stream.write(text+'\n')
        self.stream.flush()
        tmp = self.directory/'latest.tmp'
        tmp.write_text(text,encoding='utf-8')
        try:
            tmp.replace(self.directory/'latest.json')
        except PermissionError:
            # Windows reader may briefly hold the previous snapshot open.
            # The recording is authoritative; the next update retries publication.
            pass
        t = payload['state']['t']
        phase = payload['phase']['name']
        if phase != self.last_phase or t-self.last_note >= 0.2-1e-9:
            line = f"[{t:5.2f}s | {phase}] {payload['narrative']}"
            print(line,flush=True)
            self.narrative.write(line+'\n\n')
            self.narrative.flush()
            self.last_phase, self.last_note = phase,t

    def close(self):
        self.stream.close()
        self.narrative.close()
