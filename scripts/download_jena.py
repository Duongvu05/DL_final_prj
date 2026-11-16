"""Download the Jena Climate dataset and write an hourly CSV. This is only the default dataset."""

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
BAD_WIND = ["wv (m/s)", "max. wv (m/s)"]  # the raw file marks sensor faults with -9999


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/jena_hourly.csv")
    parser.add_argument("--rule", default="1h", help="pandas resample rule")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as resp:
        payload = resp.read()
    logger.info(f"downloaded {len(payload) / 1e6:.1f} MB")
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        df = pd.read_csv(io.BytesIO(zf.read(name)))  # read in memory, nothing is extracted

    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S"))
    df.index.name = "datetime"
    df[BAD_WIND] = df[BAD_WIND].mask(df[BAD_WIND] == -9999.0)
    hourly = df.resample(args.rule, closed="right", label="right").mean().ffill()
    hourly.to_csv(out)
    logger.info(f"wrote {out} with {len(hourly)} rows and {hourly.shape[1]} columns")


if __name__ == "__main__":
    main()
