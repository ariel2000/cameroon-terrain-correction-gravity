# cameroon-terrain-correction-gravity
Python implementation and comparison of terrain correction methods for gravity anomalies in Cameroon

# Comparison of Terrain Correction Methods for Gravity Anomalies

## Project Overview

This repository contains the implementation, analysis, and comparison of several terrain correction methods applied to gravity anomalies. The objective is to evaluate the numerical differences, computational performance, and domains of validity of each method, while identifying the conditions under which their results converge or diverge.

The project includes:
- preprocessing of topographic data;
- implementation of different terrain correction methods;
- numerical comparison of the results;
- quantitative error analysis;
- visualization of terrain correction maps and differences;
- discussion of the advantages and limitations of each approach.

## Research Objective

The main objective is to compare terrain correction methods used in gravimetry and determine:

- their theoretical differences;
- their numerical behavior;
- their computational efficiency;
- their sensitivity to topographic complexity;
- the conditions under which they produce similar results.

- The project currently uses original grid files stored in:
data/raw/
Processed files will be stored in:
data/processed/
Metadata and data descriptions will be stored in:
data/metadata/

## Methodological plan
The work will be carried out in the following stages:
Inspect and document the original data.
Decode and visualize the input grid.
Implement each terrain correction method independently.
Apply all methods to the same dataset.
Compare the resulting corrections statistically and spatially.
Analyze convergence and divergence between methods.
Prepare figures, tables, documentation, and manuscript material.

## Expected outputs
The repository will produce:
terrain correction maps;
comparison maps between methods;
statistical tables;
convergence plots;
reproducible Python scripts;
documentation for the scientific article.

## Authors

- **Bounou Guimking Ariel,
  Master degree in Geophysics
Novosibirsk State Technical University, Russia Federation**

- **Katchadjou Jean Rolin
 Master degree in Geophysics
  University of yaoundé 1**

## Scientific Supervisor

**Pr Kamgia Joseph**   
Research Director 
Institut National de Cartographie (INC)  
Cameroon

## Repository Structure

```
cameroon-terrain-correction-gravity/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── src/
│   ├── vannes_method/
│   ├── hammer_method/
│   ├── nagy_method/
│   ├── prism_method/
│   ├── spherical_cap_method/
│   ├── comparison/
│   └── utils/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── maps/
├── docs/
├── article/
│   ├── manuscript.md
│   ├── references.bib
│   └── figures/
├── tests/
├── README.md
└── requirements.txt
```

## Programming Language

- Python

## Status

Work in progress.
