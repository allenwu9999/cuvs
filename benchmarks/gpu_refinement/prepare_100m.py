"""Download the first 100M real BIGANN vectors as uint8; conversion is untimed in the driver."""
from pathlib import Path
from prepare_bigann_source import download

root = Path(__file__).resolve().parent / 'datasets/bigann-100m'
root.mkdir(parents=True, exist_ok=True)
print('Preparing 100,000,000 real BIGANN vectors (128 dimensions)', flush=True)
print(download(root, 100_000_000), flush=True)
