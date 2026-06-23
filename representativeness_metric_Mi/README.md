# Spatial Representativeness Metric (M_i)

Compute a per-area **spatial representativeness** score, `M_i`, for inferred home locations
derived from mobile-device / GPS data. The metric tests whether the inferred home locations
reproduce the spatial distribution expected if the sample were representative of the resident
population. It implements Section 3.4, "Detecting spatial bias by sample representativeness."

## What it computes

For each geographic unit `i` (county, census tract, or census block group), the script compares
the observed point pattern of inferred home locations against a Monte Carlo "theoretical"
pattern drawn at random over residential land:

```
M_i = exp( -| r_i / h_i - 1 | )            # bounded in [0, 1]

  r_i = AKNND(observed inferred home locations)
  h_i = AKNND(Monte Carlo points on residential land)
```

`AKNND` is the Average K-Nearest Neighbor Distance. Interpretation:

| Condition   | Meaning                                  | Score        |
|-------------|------------------------------------------|--------------|
| `r_i = h_i` | matches the theoretical distribution     | `M_i = 1`    |
| `r_i < h_i` | more clustered than theoretical          | `M_i -> 0`   |
| `r_i > h_i` | more dispersed than theoretical          | `M_i -> 0`   |

The pipeline is: count inferred home locations per area (`C_i`), draw `C_i` random points within
that area's residential polygons, compute AKNND for both the observed and simulated points, then
evaluate `M_i`. The score is computed at three scales (County, Census Tract, Census Block Group)
and mapped over Florida.

This script produces `M_i` only. Aggregating `M_i` into sampling-rate groups and computing
percentage differences by spatial-factor quantiles are downstream steps and are not included here.

## Required input data

The script uses **relative paths**. It expects to run from its own folder, with a `5_Data/`
directory one level up. Output figures are written to `4_Figure/` one level up.

```
your-repo/
├── scripts/
│   └── representativeness_metric.py        # run from inside this folder
├── 5_Data/                                 # all inputs live here (see table)
└── 4_Figure/
    └── Florida_GPS/                        # output PNGs are written here
```

All input files, with paths relative to `5_Data/`:

| Path under `5_Data/` | Description | Source | Needed for |
|---|---|---|---|
| `0_Residence_FL.csv` | Inferred home locations. Requires columns `prj_lon`, `prj_lat` (projected coordinates in EPSG:32617). | Your mobility-data processing | Core |
| `Geographic/Landuse/FL_residential.geojson` | Residential land-use polygons. | Land-use / OSM source | Core |
| `GeographicShapefile/County_US/tl_2025_us_county/tl_2025_us_county.shp` | US county boundaries. Uses `STATEFP`, `NAMELSAD`, `GEOIDFQ`. | Census TIGER/Line 2025 | Core |
| `GeographicShapefile/CensusTract/tl_2025_12_tract/tl_2025_12_tract.shp` | Florida census tracts. Uses `GEOIDFQ`. | Census TIGER/Line 2025 | Core |
| `GeographicShapefile/CensusBlockGroup/tl_2025_12_bg/tl_2025_12_bg.shp` | Florida census block groups. Uses `GEOIDFQ`. | Census TIGER/Line 2025 | Core |
| `GeographicShapefile/Boundary/US_Urban_Boundary/tl_2025_us_uac20.shp` | Urban-area boundaries. Uses `NAME20`. | Census TIGER/Line 2025 | Maps only |
| `PopulationACS/County/CountyPopACS.csv` | County population. Uses `GEO_ID`, `B01003_001E`. | data.census.gov, ACS 5-year table B01003 | Loaded, not used by `M_i` |
| `PopulationACS/CensusTract/CensusTractPopACS.csv` | Census-tract population. Uses `GEO_ID`, `B01003_001E`. | ACS 5-year table B01003 | Loaded, not used by `M_i` |
| `PopulationACS/CensusBlockGroup/CensusBlockGroupPopACS.csv` | Block-group population. Uses `GEO_ID`, `B01003_001E`. | ACS 5-year table B01003 | Loaded, not used by `M_i` |
| `EmploymentACS/County/County/ACSST5Y2024.S2301-Data.csv` | County employment. Uses `GEO_ID`, `S2301_C01_001E`, `S2301_C03_001E`. | ACS 5-year subject table S2301 | Loaded, not used by `M_i` |
| `EmploymentACS/CensusTract/CensusTract/ACSST5Y2024.S2301-Data.csv` | Census-tract employment. Same columns as above. | ACS 5-year subject table S2301 | Loaded, not used by `M_i` |
| `EmploymentACS/CensusBlockGroup/CensusBlockGroup/ACSDT5Y2024.B23025-Data.csv` | Block-group employment. Uses `GEO_ID`, `B23025_002E`. | ACS 5-year detailed table B23025 | Loaded, not used by `M_i` |

Notes:

- **Shapefiles need their sidecar files.** Each `.shp` must keep its companions (`.shx`, `.dbf`,
  `.prj`, and `.cpg` if present) in the same folder.
- **"Loaded, not used by `M_i`."** The data-loading cell reads the ACS population and employment
  files and attaches them as attributes, but the `M_i` computation and the maps do not use them.
  They support the downstream sampling-rate analysis. The script will still error if these files
  are missing. To run `M_i` without them, comment out the corresponding lines in the data-loading
  cell. The ACS population/employment files are otherwise optional for this script.
- **"Maps only."** The urban-boundary shapefile is used only as an overlay in the final map. Skip it
  if you do not run the plotting cell.

## Output

Three maps of `M_i` over Florida, written to `4_Figure/Florida_GPS/`:

- `FL_county_M_i_mean.png`
- `FL_CT_M_i_mean.png`
- `FL_CBG_M_i_mean.png`

The per-unit results are also held in memory as `results_county_df`, `results_CT_df`, and
`results_CBG_df` (columns: `Ci`, `r_i`, `h_i_mean`, `M_i_mean`, plus the area id).

## Requirements

Python 3.9 or later, with:

```bash
pip install pandas geopandas numpy shapely matplotlib contextily scikit-learn tqdm
```

The plotting cell calls `contextily`, which downloads basemap tiles, so that cell needs an
internet connection.

## Running

The script resolves paths relative to the working directory, so run it from its own folder:

```bash
cd scripts
python representativeness_metric.py
```

Alternatively, open it in VS Code or Jupyter and run it cell by cell. The `# %%` markers define the
cells and mirror the original notebook.

**For final results, set `n_sim = 100`.** The default is `2`, a fast placeholder for development.
A two-run mean is unstable and cannot support the hypothesis test described in the paper.

## Key parameters

All human-set parameters are documented in the header of `representativeness_metric.py`:

- `k = 5` — number of nearest neighbors for AKNND.
- `n_sim = 2` — Monte Carlo simulations per unit. Set to `100` for the paper.
- `EPSG:32617` — projected CRS (UTM zone 17N, meters). Change to the matching UTM zone for a
  different study area.
- Random seed — not set by default. Add `np.random.seed(...)` and `random.seed(...)` for
  reproducible runs.

## Data availability

Input data are **not** included in this repository. Mobile-device / GPS data and any home
locations derived from them are subject to their providers' licenses and are not redistributed
here. The public inputs (Census TIGER/Line boundaries and ACS tables) are available from the
sources listed above. The repository `.gitignore` excludes `5_Data/` and `4_Figure/`.
