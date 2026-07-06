# Data Description

## Project

**Cameroon Terrain Correction Gravity**

Comparative study of terrain correction methods applied to gravity anomalies.

---

# Dataset overview

The project uses two complementary datasets.

## 1. données orig1.dat

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

## 2. out1.dat

### Description

This file represents the complete study area as a regular grid.

The grid was generated from the original observation points (`données orig1.dat`) using **Ordinary Kriging** in **Golden Software Surfer**.

The interpolation was performed exclusively for visualization and spatial representation purposes.

### Purpose

- visualization of the entire study area;
- production of maps;
- qualitative comparison of terrain correction methods;
- spatial interpretation of the results.

### Important note

The interpolated grid **is not used as the reference dataset for numerical comparisons**.

All terrain correction computations are performed only at the **1991 original observation stations** contained in `données orig1.dat`.

---

# Data workflow

Original field stations
↓
données orig1.dat (1991 stations)
↓
Ordinary Kriging (Surfer)
↓
out1.dat (continuous grid)
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
