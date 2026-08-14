from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_CSV = Path("khaki_multi_crop_yield.csv")
OUTPUT_DIR = Path("data/cornbelt")

TRAIN_YEARS = [2013, 2014, 2015, 2016]
VAL_YEAR = 2017
TEST_YEAR = 2018


# Public dataset weather mapping
WEATHER_MAP = {
    "prcp": 1,
    "srad": 2,
    "swe": 3,
    "tmax": 4,
    "tmin": 5,
    "vp": 6,
}


SOIL_MEASUREMENTS = [
    "bdod",
    "cec",
    "cfvo",
    "clay",
    "nitrogen",
    "ocd",
    "ocs",
    "phh2o",
    "sand",
    "silt",
    "soc",
]


SOIL_DEPTHS = [
    "0-5cm",
    "5-15cm",
    "15-30cm",
    "30-60cm",
    "60-100cm",
    "100-200cm",
]


SOIL_VARS = [
    f"{measurement}_mean_{depth}"
    for measurement in SOIL_MEASUREMENTS
    for depth in SOIL_DEPTHS
]


# ============================================================
# Load
# ============================================================

print(f"Reading {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

print("Raw shape:", df.shape)


# ============================================================
# Verify required columns
# ============================================================

if "corn_yield" not in df.columns:
    raise ValueError("corn_yield column not found")


required_meta = [
    "loc_ID",
    "year",
    "State",
    "County",
    "lat",
    "lng",
]

missing_meta = [
    c for c in required_meta
    if c not in df.columns
]

if missing_meta:
    raise ValueError(
        f"Missing metadata columns: {missing_meta}"
    )


missing_soil = [
    c for c in SOIL_VARS
    if c not in df.columns
]

if missing_soil:
    raise ValueError(
        f"Missing soil columns: {missing_soil}"
    )


# Verify all 52 weekly values exist for each weather variable
missing_weather = []

for weather_name, source_idx in WEATHER_MAP.items():
    for week in range(1, 53):
        src = f"W_{source_idx}_{week}"

        if src not in df.columns:
            missing_weather.append(src)

if missing_weather:
    raise ValueError(
        f"Missing weather columns: {missing_weather[:20]}"
    )


# ============================================================
# Convert year and corn yield to numeric
# ============================================================

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce",
)

df["corn_yield"] = pd.to_numeric(
    df["corn_yield"],
    errors="coerce",
)


# ============================================================
# IMPORTANT:
# Filter to corn samples from 2013-2018 BEFORE constructing out
# ============================================================

df = df[
    (df["year"] >= 2013) &
    (df["year"] <= 2018)
].copy()


# Remove samples without valid corn yield or year
df = df.dropna(
    subset=[
        "corn_yield",
        "year",
    ]
).reset_index(drop=True)


df["year"] = df["year"].astype(int)


print()
print("===================================")
print("FILTERED SOURCE DATA")
print("===================================")

print(
    "Years retained:",
    sorted(df["year"].unique())
)

print(
    "Rows after 2013-2018 + corn yield filter:",
    len(df)
)

print("\nSamples per year:")
print(
    df["year"]
    .value_counts()
    .sort_index()
)


expected_years = {
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
}

assert set(df["year"].unique()) == expected_years


# ============================================================
# Construct metadata block
# ============================================================

metadata_df = pd.DataFrame(
    {
        "crop": "corn",

        # County/location acts as the sample spatial identifier.
        # It is metadata required by the current dataset interface.
        "farm_field": (
            "county_" +
            df["loc_ID"].astype(str)
        ),

        "year": df["year"].astype(int),

        "yield": df[
            "corn_yield"
        ].astype(np.float32),

        "loc_ID": df["loc_ID"],

        "state": df[
            "State"
        ].astype(str),

        "county": df[
            "County"
        ].astype(str),

        "lat": pd.to_numeric(
            df["lat"],
            errors="coerce",
        ).astype(np.float32),

        "lng": pd.to_numeric(
            df["lng"],
            errors="coerce",
        ).astype(np.float32),
    }
)


# ============================================================
# Weather block
#
# IMPORTANT:
# - No interpolation
# - No daily conversion
# - No aggregation
#
# Each source W_x_1 ... W_x_52 is copied directly.
#
# Example:
# W_1_1 -> prcp_0
# W_1_2 -> prcp_1
# ...
# W_1_52 -> prcp_51
# ============================================================

weather_data = {}


for weather_name, source_idx in WEATHER_MAP.items():

    for week in range(1, 53):

        src = f"W_{source_idx}_{week}"

        # Zero-based temporal indexing used by current loader
        dst = f"{weather_name}_{week - 1}"

        weather_data[dst] = pd.to_numeric(
            df[src],
            errors="coerce",
        ).astype(np.float32)


weather_df = pd.DataFrame(
    weather_data,
    index=df.index,
)


# ============================================================
# Soil block
# ============================================================

soil_data = {}


for col in SOIL_VARS:

    soil_data[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    ).astype(np.float32)


soil_df = pd.DataFrame(
    soil_data,
    index=df.index,
)


# ============================================================
# Combine all blocks
# ============================================================

out = pd.concat(
    [
        metadata_df.reset_index(drop=True),
        weather_df.reset_index(drop=True),
        soil_df.reset_index(drop=True),
    ],
    axis=1,
)


# Replace inf values with NaN.
# Normalization / missing-value handling remains in train_hf.py
# and YieldDataset.
out = out.replace(
    [np.inf, -np.inf],
    np.nan,
)


# ============================================================
# Critical safety check
# ============================================================

print()
print("===================================")
print("FINAL MODEL DATASET")
print("===================================")

print(
    "Shape:",
    out.shape,
)

print(
    "Years:",
    sorted(out["year"].unique())
)

print(
    "Rows:",
    len(out)
)


assert set(out["year"].unique()) == {
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
}


assert len(out) == len(df)


# ============================================================
# Chronological split
#
# Train: 2013-2016
# Val:   2017
# Test:  2018
# ============================================================

train_df = out[
    out["year"].isin(TRAIN_YEARS)
].reset_index(drop=True)


val_df = out[
    out["year"] == VAL_YEAR
].reset_index(drop=True)


test_df = out[
    out["year"] == TEST_YEAR
].reset_index(drop=True)


# ============================================================
# Split reporting
# ============================================================

print()
print("===================================")
print("SPLITS")
print("===================================")


print("\nTRAIN")

print(
    train_df[
        "year"
    ].value_counts().sort_index()
)

print(
    "Samples:",
    len(train_df),
)

print(
    "Counties:",
    train_df[
        "farm_field"
    ].nunique(),
)


print("\nVALIDATION")

print(
    val_df[
        "year"
    ].value_counts().sort_index()
)

print(
    "Samples:",
    len(val_df),
)

print(
    "Counties:",
    val_df[
        "farm_field"
    ].nunique(),
)


print("\nTEST")

print(
    test_df[
        "year"
    ].value_counts().sort_index()
)

print(
    "Samples:",
    len(test_df),
)

print(
    "Counties:",
    test_df[
        "farm_field"
    ].nunique(),
)


# ============================================================
# Split sanity checks
# ============================================================

assert set(
    train_df["year"].unique()
) == {
    2013,
    2014,
    2015,
    2016,
}


assert set(
    val_df["year"].unique()
) == {
    2017,
}


assert set(
    test_df["year"].unique()
) == {
    2018,
}


assert (
    len(train_df)
    + len(val_df)
    + len(test_df)
    == len(out)
)


# No row should appear in more than one split
assert set(train_df.index).isdisjoint(
    set(range(
        len(train_df),
        len(train_df) + len(val_df)
    ))
)


print()
print("All year/split checks passed.")


# ============================================================
# Weather sanity checks
# ============================================================

weather_vars = list(
    WEATHER_MAP.keys()
)


for weather_var in weather_vars:

    cols = [
        c
        for c in out.columns
        if c.startswith(
            f"{weather_var}_"
        )
    ]

    assert len(cols) == 52, (
        f"{weather_var}: expected 52 weekly "
        f"columns, found {len(cols)}"
    )


print()
print("Weather variables:")
print(weather_vars)

print(
    "Weeks per weather variable:",
    52,
)

print(
    "Total weather columns:",
    52 * len(weather_vars),
)


# ============================================================
# Soil sanity checks
# ============================================================

assert len(SOIL_VARS) == 66

assert all(
    col in out.columns
    for col in SOIL_VARS
)


print(
    "Number of soil variables:",
    len(SOIL_VARS),
)


# ============================================================
# Check expected tensor dimensions
# ============================================================

print()
print("Expected model input dimensions:")

print(
    "weather = [52, 6]"
)

print(
    "soil    = [66]"
)


# ============================================================
# Yield summaries
# ============================================================

print()
print("===================================")
print("YIELD SUMMARY")
print("===================================")


print("\nTrain:")
print(
    train_df[
        "yield"
    ].describe()
)


print("\nValidation:")
print(
    val_df[
        "yield"
    ].describe()
)


print("\nTest:")
print(
    test_df[
        "yield"
    ].describe()
)


# ============================================================
# Save HDF5 files
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


train_path = OUTPUT_DIR / "train.h5"
val_path = OUTPUT_DIR / "val.h5"
test_path = OUTPUT_DIR / "test.h5"


train_df.to_hdf(
    train_path,
    key="data",
    mode="w",
)


val_df.to_hdf(
    val_path,
    key="data",
    mode="w",
)


test_df.to_hdf(
    test_path,
    key="data",
    mode="w",
)


print()
print("===================================")
print("SAVED")
print("===================================")

print(train_path)
print(val_path)
print(test_path)


# ============================================================
# Reload files to verify they were written correctly
# ============================================================

train_check = pd.read_hdf(train_path)
val_check = pd.read_hdf(val_path)
test_check = pd.read_hdf(test_path)


assert len(train_check) == len(train_df)
assert len(val_check) == len(val_df)
assert len(test_check) == len(test_df)


assert set(
    train_check["year"].unique()
) == {
    2013,
    2014,
    2015,
    2016,
}


assert set(
    val_check["year"].unique()
) == {
    2017,
}


assert set(
    test_check["year"].unique()
) == {
    2018,
}


print()
print("HDF5 reload verification passed.")

print()
print("Done.")

