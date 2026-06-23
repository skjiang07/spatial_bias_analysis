# -*- coding: utf-8 -*-
"""
================================================================================
Spatial Representativeness Metric (M_i)
================================================================================

Metric:
        M_i = exp( -| r_i / h_i - 1 | )                  # bounded in [0, 1]
    where
        r_i = AKNND(empirical / inferred home locations)   # observed distribution
        h_i = AKNND(Monte Carlo simulated distribution)    # random points on residential land
    Interpretation:
        r_i = h_i  ->  M_i = 1   fully representative
        r_i < h_i  ->  more clustered than the theoretical distribution
        r_i > h_i  ->  more dispersed than the theoretical distribution
        M_i = 0    ->  poorly representative

Pipeline:
    Inferred home locations + residential land use  ->  count C_i per area
    ->  Monte Carlo: draw C_i random points within residential polygons (theoretical distribution)
    ->  compute AKNND for the empirical and simulated points  ->  r_i, h_i  ->  M_i

Scope:
    This script computes M_i per geographic unit only. Aggregating M_i into
    sampling-rate groups (M_g) and computing percentage differences by spatial-factor
    quantiles (Delta M_g,q,alpha) are downstream steps and are not included here.

--------------------------------------------------------------------------------
HUMAN-SET PARAMETERS (changing any of these changes the results):
--------------------------------------------------------------------------------
  * k = 5
        Number of nearest neighbors used in AKNND. It appears as a default in both
        compute_aknnd() and evaluate_unit(); keep the two consistent. Set it to the
        value reported in the paper.

  * n_sim = 2
        Number of Monte Carlo simulations per geographic unit.

  * CRS = EPSG:32617  (UTM zone 17N, meters)
        The empirical points and all geographic and residential layers are projected to
        this CRS, so AKNND distances are in meters. Florida falls in zone 17N; use the
        matching UTM zone for a different study area. M_i is a ratio (r_i / h_i) and is
        therefore unit-invariant, but both terms must share the same CRS.

  * STATEFP == '12'
        State FIPS code for Florida, used to filter the national datasets.

  * Random seed (not set)
        No seed is fixed, so results vary across runs, more so when n_sim is small.
        For reproducibility, add at the top of the script:
            np.random.seed(42); random.seed(42)
        The original code does not set a seed; this is a note only.
================================================================================
"""

# %% [cell 0] Dependencies
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import contextily as ctx                          # basemap for the final maps
from sklearn.neighbors import NearestNeighbors    # nearest-neighbor search for AKNND
import random                                     # random sampling of points within polygons
from tqdm import tqdm                             # progress bar
import warnings
warnings.filterwarnings("ignore")


# %% [cell 1] Load empirical data: inferred home locations (source of the observed distribution)
resident_FL = pd.read_csv('../5_Data/0_Residence_FL.csv')

resident_FL_gdf = gpd.GeoDataFrame(
    resident_FL,
    geometry=gpd.points_from_xy(
        resident_FL['prj_lon'],  # x (projected longitude)
        resident_FL['prj_lat']   # y (projected latitude)
    ),
    # HUMAN-SET: CRS = EPSG:32617 (UTM 17N, meters). This sets the CRS of the empirical
    # points; all layers below are projected to the same CRS.
    crs="EPSG:32617"
)


# %% [cell 2] Load residential land use (quick check only; the projected version in cell 7 is the one used)
residential_area = gpd.read_file('../5_Data/Geographic/Landuse/FL_residential.geojson')


# %% [cell 3] Three geographic boundary sets plus ACS population/employment (County / Census Tract / CBG)
######################## County ########################

US_county = gpd.read_file('../5_Data/GeographicShapefile/County_US/tl_2025_us_county/tl_2025_us_county.shp')
# HUMAN-SET: STATEFP == '12' selects Florida; project to EPSG:32617 (meters).
FL_county = US_county[US_county['STATEFP'] == '12'].to_crs(epsg=32617)

# ACS total population (B01003_001E). Population feeds the downstream sampling-rate and
# grouping analysis. It does not set the number of simulated points in M_i (that number
# equals the empirical count C_i; see evaluate_unit).
PopCounty = pd.read_csv('../5_Data/PopulationACS/County/CountyPopACS.csv').iloc[1:].reset_index(drop=True)
FL_county["ACSPopCounty"] = FL_county["GEOIDFQ"].map(PopCounty.set_index("GEO_ID")["B01003_001E"])

# ACS employed population = labor force total * employment rate / 100
EmpCounty = pd.read_csv('../5_Data/EmploymentACS/County/County/ACSST5Y2024.S2301-Data.csv').iloc[1:].reset_index(drop=True)
EmpCounty["EmpPop"] = pd.to_numeric(EmpCounty["S2301_C01_001E"], errors="coerce") * pd.to_numeric(EmpCounty["S2301_C03_001E"], errors="coerce") / 100
FL_county["EmpPopCounty"] = FL_county["GEOIDFQ"].map(EmpCounty.set_index("GEO_ID")["EmpPop"])

######################## Census tract ########################

FL_CT = gpd.read_file('../5_Data/GeographicShapefile/CensusTract/tl_2025_12_tract/tl_2025_12_tract.shp').to_crs(epsg=32617)

PopCT = pd.read_csv('../5_Data/PopulationACS/CensusTract/CensusTractPopACS.csv').iloc[1:].reset_index(drop=True)
FL_CT["ACSPopCT"] = FL_CT["GEOIDFQ"].map(PopCT.set_index("GEO_ID")["B01003_001E"])

EmpCT = pd.read_csv('../5_Data/EmploymentACS/CensusTract/CensusTract/ACSST5Y2024.S2301-Data.csv').iloc[1:].reset_index(drop=True)
EmpCT["EmpPop"] = pd.to_numeric(EmpCT["S2301_C01_001E"], errors="coerce") * pd.to_numeric(EmpCT["S2301_C03_001E"], errors="coerce") / 100
FL_CT["EmpPopCT"] = FL_CT["GEOIDFQ"].map(EmpCT.set_index("GEO_ID")["EmpPop"])

######################## Census block group ########################

FL_CBG = gpd.read_file('../5_Data/GeographicShapefile/CensusBlockGroup/tl_2025_12_bg/tl_2025_12_bg.shp').to_crs(epsg=32617)

PopCBG = pd.read_csv('../5_Data/PopulationACS/CensusBlockGroup/CensusBlockGroupPopACS.csv').iloc[1:].reset_index(drop=True)
FL_CBG["ACSPopCBG"] = FL_CBG["GEOIDFQ"].map(PopCBG.set_index("GEO_ID")["B01003_001E"])

EmpCBG = pd.read_csv('../5_Data/EmploymentACS/CensusBlockGroup/CensusBlockGroup/ACSDT5Y2024.B23025-Data.csv').iloc[1:].reset_index(drop=True)
FL_CBG["EmpPopCBG"] = FL_CBG["GEOIDFQ"].map(EmpCBG.set_index("GEO_ID")["B23025_002E"])


# %% [cell 4] Tag each empirical point with its County / CT / CBG (spatial join, point in polygon)
# -------------------------
# County
# -------------------------
resident_FL_gdf = gpd.sjoin(
    resident_FL_gdf,
    FL_county[["NAMELSAD", "geometry"]],
    how="left",
    predicate="within"           # whether the point falls inside the polygon
).rename(columns={"NAMELSAD": "County"}).drop(columns=["index_right"])


# -------------------------
# Census Tract
# -------------------------
resident_FL_gdf = gpd.sjoin(
    resident_FL_gdf,
    FL_CT[["GEOIDFQ", "geometry"]],
    how="left",
    predicate="within"
).rename(columns={"GEOIDFQ": "CT"}).drop(columns=["index_right"])


# -------------------------
# Census Block Group
# -------------------------
resident_FL_gdf = gpd.sjoin(
    resident_FL_gdf,
    FL_CBG[["GEOIDFQ", "geometry"]],
    how="left",
    predicate="within"
).rename(columns={"GEOIDFQ": "CBG"}).drop(columns=["index_right"])


# %% [cell 5] Urban boundaries (used only as a map overlay at the end)
US_urban = gpd.read_file('../5_Data/GeographicShapefile/Boundary/US_Urban_Boundary/tl_2025_us_uac20.shp').to_crs(epsg=32617)
FL_urban = US_urban[
    US_urban["NAME20"].str.contains("FL", na=False)
]


# %% [markdown] ## Residential area: polygons used to draw random points on residential land
# %% [cell 7] Residential polygons actually used (projected to EPSG:32617, meters)
residential_area_FL = gpd.read_file('../5_Data/Geographic/Landuse/FL_residential.geojson').to_crs(epsg=32617)


# %% [cell 8] Quick check of the residential polygons (rows, columns)
gpd.read_file('../5_Data/Geographic/Landuse/FL_residential.geojson').shape


# %% [cell 9] Clip residential polygons by each boundary set (overlay intersection) to get residential sub-polygons per unit
residential_county = gpd.overlay(residential_area_FL, FL_county[["NAMELSAD", "geometry"]], how="intersection")
residential_county = residential_county.rename(columns={"NAMELSAD": "County"})

residential_CT = gpd.overlay(residential_area_FL, FL_CT[["GEOIDFQ", "geometry"]], how="intersection")
residential_CT = residential_CT.rename(columns={"GEOIDFQ": "CT"})

residential_CBG = gpd.overlay(residential_area_FL, FL_CBG[["GEOIDFQ", "geometry"]], how="intersection")
residential_CBG = residential_CBG.rename(columns={"GEOIDFQ": "CBG"})


# %% [cell 10] Check the number of residential sub-polygons at each scale
print(residential_county.shape)
print(residential_CT.shape)
print(residential_CBG.shape)


# %% [markdown] ## Represent metric: the core computation (the four functions below are the metric itself)
# %% [cell 12] Function 1: compute AKNND (Average K-Nearest Neighbor Distance)
def compute_aknnd(points_gdf, k=5):
    """
    Average K-nearest-neighbor distance (AKNND). Measures the clustering or dispersion of a point set.
    HUMAN-SET k: number of nearest neighbors per point (default 5; keep consistent with evaluate_unit).
    """
    # Extract (x, y) for every point (CRS=32617, meters, so distances are in meters)
    coords = np.array([[geom.x, geom.y] for geom in points_gdf.geometry])

    # Return NaN when there are too few points to find k neighbors (small units are skipped)
    if len(coords) <= k:
        return np.nan

    # n_neighbors = k+1 because the nearest (0th) neighbor of a point is the point itself (distance 0)
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(coords)

    distances, _ = nn.kneighbors(coords)

    # Column 0 is the point itself; drop it and keep columns 1..k as the true k neighbors
    knn = distances[:, 1:k+1]

    # Averaging over all points and their k neighbors gives AKNND
    return knn.mean()


# %% [cell 13] Function 2: draw n random points within a single polygon (rejection sampling)
def random_points_in_polygon(polygon, n):
    """
    Draw n points uniformly at random within one polygon.
    Sample within the bounding box and keep a point only if it falls inside the polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds   # bounding box

    points = []

    while len(points) < n:

        p = Point(
            random.uniform(minx, maxx),
            random.uniform(miny, maxy)
        )

        if polygon.contains(p):   # keep only points that fall inside the polygon
            points.append(p)

    return points


# %% [cell 14] Function 3: draw points across multiple residential polygons, weighted by area
def random_points_in_polygons(polygons_gdf, n):
    """
    Draw n points in total across the residential sub-polygons of one geographic unit.
    Area weighting assigns more points to larger sub-polygons, which approximates a
    uniform distribution over residential land.
    """
    polygons = polygons_gdf.copy()

    polygons["area"] = polygons.geometry.area              # area of each sub-polygon

    probs = polygons["area"] / polygons["area"].sum()      # normalize to sampling probabilities

    # Sample n times by area probability. np.random.choice draws with replacement, so a large
    # polygon can be selected multiple times and hold multiple points; this is intended.
    chosen = np.random.choice(polygons.index, size=n, p=probs)

    pts = []

    for idx in chosen:

        poly = polygons.loc[idx, "geometry"]

        pts.extend(random_points_in_polygon(poly, 1))      # one point each time a polygon is selected

    return pts


# %% [cell 15] Function 4: compute M_i for one geographic unit (core of the metric)
def evaluate_unit(emp_points, res_polygons, k=5, n_sim=2):
    """
    Compute M_i for a single geographic unit.

    HUMAN-SET parameters:
      k     : K for AKNND (default 5; must match compute_aknnd).
      n_sim : number of Monte Carlo simulations.
    """
    Ci = len(emp_points)   # C_i: number of inferred home locations in this unit

    # Skip the unit if it has too few points for AKNND or no residential polygons
    if Ci <= k or len(res_polygons) == 0:
        return None

    # r_i: AKNND of the empirical (observed) points
    r_i = compute_aknnd(emp_points, k)

    h_vals = []   # h_i from each simulation
    M_vals = []   # M_i from each simulation

    # Repeat n_sim Monte Carlo simulations. The theoretical distribution is random, so average over runs.
    for _ in range(n_sim):

        # Draw C_i random points within the residential polygons. This is the theoretical distribution.
        # The number of points equals C_i (the empirical count), not ACS population, matching the paper text.
        sim_points = random_points_in_polygons(res_polygons, Ci)

        sim_gdf = gpd.GeoDataFrame(
            geometry=sim_points,
            crs=emp_points.crs        # same CRS as the empirical points so r_i and h_i share units
        )

        # h_i: AKNND of the simulated (theoretical) points
        h_i = compute_aknnd(sim_gdf, k)

        if pd.notna(h_i):

            h_vals.append(h_i)

            # Core formula: M_i = exp( -| r_i / h_i - 1 | )
            M = np.exp(-abs(r_i / h_i - 1))

            M_vals.append(M)

    # Return point estimates, averaged over n_sim runs.
    # The hypothesis test in the paper (based on 100 runs) is not implemented here;
    # h_vals holds the full distribution if the test is added externally.
    return {
        "Ci": Ci,
        "r_i": r_i,
        "h_i_mean": np.mean(h_vals),
        "M_i_mean": np.mean(M_vals)
    }


# %% [cell 16] Apply at the County scale
# evaluate_unit(emp, res) does not pass n_sim or k, so it uses the defaults n_sim=2, k=5
results = []

counties = residential_county["County"].unique()

for county in tqdm(counties, desc="Processing counties"):

    # empirical points in this county
    emp = resident_FL_gdf[
        resident_FL_gdf["County"] == county
    ]

    # residential sub-polygons in this county
    res = residential_county[
        residential_county["County"] == county
    ]

    out = evaluate_unit(emp, res)

    if out is not None:

        out["County"] = county

        results.append(out)


# %% [cell 17] Inspect county-level residential sub-polygons
residential_county


# %% [cell 18] Convert county-level results to a DataFrame
results_county_df = pd.DataFrame(results)


# %% [cell 19] Apply at the Census Tract scale (same defaults n_sim=2, k=5)
results_CT = []

cts = residential_CT["CT"].unique()

for ct in tqdm(cts, desc="Processing CT"):

    emp = resident_FL_gdf[
        resident_FL_gdf["CT"] == ct
    ]

    res = residential_CT[
        residential_CT["CT"] == ct
    ]

    out = evaluate_unit(emp, res)

    if out is not None:

        out["CT"] = ct

        results_CT.append(out)


# %% [cell 20] Convert CT-level results to a DataFrame
results_CT_df = pd.DataFrame(results_CT)


# %% [cell 21] (Empty cell in the original notebook; kept as a placeholder)


# %% [cell 22] Apply at the Census Block Group scale (same defaults n_sim=2, k=5)
results_CBG = []

cbgs = residential_CBG["CBG"].unique()

for cbg in tqdm(cbgs, desc="Processing CBG"):

    emp = resident_FL_gdf[
        resident_FL_gdf["CBG"] == cbg
    ]

    res = residential_CBG[
        residential_CBG["CBG"] == cbg
    ]

    out = evaluate_unit(emp, res)

    if out is not None:

        out["CBG"] = cbg

        results_CBG.append(out)

results_CBG_df = pd.DataFrame(results_CBG)


# %% [cell 23] Descriptive statistics of M_i at the CBG scale
results_CBG_df['M_i_mean'].describe()


# %% [cell 24] Merge results back to geometry for mapping
def merge_geometry(df, df_key, geo_df, geo_key):
    """Join results to their geometry by key and return a GeoDataFrame."""
    merged = df.merge(geo_df[[geo_key, 'geometry']], left_on=df_key, right_on=geo_key, how='left')
    if df_key != geo_key:
        merged = merged.drop(columns=geo_key)
    return gpd.GeoDataFrame(merged, geometry='geometry', crs=geo_df.crs)

results_county_gdf = merge_geometry(results_county_df, 'County', FL_county, 'NAMELSAD')
results_CT_gdf     = merge_geometry(results_CT_df, 'CT',     FL_CT,     'GEOIDFQ')
results_CBG_gdf    = merge_geometry(results_CBG_df, 'CBG',   FL_CBG,    'GEOIDFQ')


# %% [cell 25] Inspect CBG-level results with geometry
results_CBG_gdf


# %% [cell 26] Plot the spatial distribution of M_i at the three scales
def plot_M_i_mean(gdf, title, save_path=None, urban_gdf=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.to_crs(epsg=3857).plot(             # reproject to Web Mercator for the basemap
        column="M_i_mean",
        cmap="OrRd",                        # HUMAN-SET: color scheme
        alpha=0.8,
        legend=True,
        legend_kwds={"label": "M_i_mean", "shrink": 0.6},
        missing_kwds={"color": "lightgrey", "label": "No Data"},   # units with no data shown in grey
        # edgecolor="white",
        linewidth=0.4,
        ax=ax
    )
    if urban_gdf is not None:               # overlay urban boundaries (black outline)
        urban_gdf.to_crs(epsg=3857).plot(
            ax=ax, facecolor="none", edgecolor="black", linewidth=0.5
        )
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Plot the County / CT / CBG scales
plot_M_i_mean(results_county_gdf, "Florida County-Level Spatial representativeness",
              save_path="../4_Figure/Florida_GPS/FL_county_M_i_mean.png", urban_gdf=FL_urban)
plot_M_i_mean(results_CT_gdf,     "Florida Census Tract-Level Spatial representativeness",
              save_path="../4_Figure/Florida_GPS/FL_CT_M_i_mean.png",     urban_gdf=FL_urban)
plot_M_i_mean(results_CBG_gdf,    "Florida CBG-Level Spatial representativeness",
              save_path="../4_Figure/Florida_GPS/FL_CBG_M_i_mean.png",    urban_gdf=FL_urban)
