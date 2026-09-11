#!/usr/bin/env python3
"""Download a checked prefix of the real BIGANN uint8 database.

Adapted from the local kuaishou-cuvs-benchmarks data preparation helper.
No synthetic data, queries, or ground truth are generated.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.request

URL = 'https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin'
CHUNK = 64 * 1024 * 1024


def download(root, rows):
    size = 8 + rows * 128
    target = root / f'base.{rows // 1000000}M.u8bin'
    if target.exists():
        with target.open('rb') as f:
            if target.stat().st_size != size or f.read(8) != struct.pack('<II', rows, 128):
                raise ValueError(f'Invalid existing dataset: {target}')
        return target
    partial = target.with_suffix(target.suffix + '.partial')
    parts = root / f'download-parts-{rows}'
    parts.mkdir(exist_ok=True)
    resume = partial.exists()
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)

    def fetch(i):
        start, end = i * CHUNK, min(size, (i+1)*CHUNK)-1
        marker = parts / f'{i:04d}.json'
        if resume and marker.exists():
            record = json.loads(marker.read_text())
            if hashlib.sha256(os.pread(fd, end-start+1, start)).hexdigest() == record['sha256']:
                return record
        for attempt in range(5):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206 or response.headers['Content-Range'] != f'bytes {start}-{end}/128000000008':
                        raise ValueError('Server did not return the requested BIGANN byte range')
                    data, etag = response.read(), response.headers['ETag']
                if len(data) != end-start+1:
                    raise ValueError('Truncated download')
                written = 0
                while written < len(data):
                    written += os.pwrite(fd, data[written:], start+written)
                record = dict(part=i, sha256=hashlib.sha256(data).hexdigest(), etag=etag)
                marker.write_text(json.dumps(record)+'\n')
                return record
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
    try:
        with ThreadPoolExecutor(8) as pool:
            records = list(pool.map(fetch, range((size+CHUNK-1)//CHUNK)))
        if len({r['etag'] for r in records}) != 1 or os.pread(fd, 8, 0) != struct.pack('<II', 1000000000, 128):
            raise ValueError('BIGANN source changed or has an invalid header')
        os.pwrite(fd, struct.pack('<II', rows, 128), 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    partial.replace(target)
    (root / f'download-{rows}.json').write_text(json.dumps(dict(url=URL, rows=rows, chunks=records), indent=2)+'\n')
    return target
