# Data Description

## Project

**Cameroon Terrain Correction Gravity**

Comparative study of terrain correction methods applied to gravity anomalies.

---

# Dataset overview

The project uses two complementary datasets.

## 1. original_stations.xlsx

### Description

This file contains the original observation points.

These are the **1991 computation stations** where terrain corrections will be calculated using the different methods implemented in this project.

Each point corresponds to an observation location acquired during the gravity survey.

These stations constitute the reference dataset for all numerical comparisons.

### Purpose

- computation of terrain corrections;
- comparison of correction methods;
- statistical analysis of the results.

---

## 2. kriged_grid_surfer.dat

### Description

This file represents the complete study area as a regular grid.

The grid was generated from the original observation points (`original_stations.xlsx`) using **Ordinary Kriging** in **Golden Software Surfer**.

The interpolation was performed exclusively for visualization and spatial representation purposes.

This file is the gridded representation of the study area.

It was obtained by kriging in Surfer from the original station data.

It is used to represent the whole area and to locate each station inside the regional terrain grid before extracting the surrounding topography.

## original_stations.xlsx

This file contains the 1991 original gravity stations.

Columns:

- Longitude: in degrees
- Latitude: in degrees
- Altitude: in meters
- A(air libre): free-air anomaly, probably in mGal

These points are the real computation points where terrain corrections will be calculated.


### Purpose

- visualization of the entire study area;
- production of maps;
- qualitative comparison of terrain correction methods;
- spatial interpretation of the results.

### Important note

The interpolated grid **is not used as the reference dataset for numerical comparisons**.

All terrain correction computations are performed only at the **1991 original observation stations** contained in `original_stations.xlsx`.

---

# Data workflow

Original field stations
↓
original_stations.xlsx (1991 stations)
↓
Ordinary Kriging (Surfer)
↓
original_stations.xlsx (continuous grid)
↓
Maps and visualization

---

# Computational workflow

Terrain correction methods

- Hammer
- Nagy
- Prism
- Spherical Cap
- Vannes

↓

Terrain corrections computed at the 1991 original stations

↓

Comparison of methods

↓

Statistical analysis

↓

Maps generated using the kriged grid.
