# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a seismic facies classification research project for the Bryne Formation (North Sea). It uses unsupervised machine learning (Self-Organizing Maps) to classify 3D seismic attribute volumes.

## Environment

No formal package manager config exists. Key dependencies:
- `segysak` — SEG-Y file I/O and xarray-based seismic data handling
- `minisom` — Self-Organizing Map implementation
- `xarray`, `numpy`, `pandas`
- `scikit-learn` (MinMaxScaler, PCA)
- `matplotlib`, `seaborn`
- `bruges` — geophysical utilities

Run notebooks with Jupyter: `jupyter notebook` or `jupyter lab`

## Architecture & Data Flow

The project is a sequential pipeline of 4 notebooks (no Python packages):

1. **`001_load_Petrel_seismic_attribute_segys.ipynb`** — Reads 13 SEG-Y attribute volumes exported from Petrel, converts to xarray Datasets, saves as `.nc` (NetCDF) files per attribute.

2. **`001_merge_attribute_volumes.ipynb`** — Merges all `.nc` attribute files and horizon surfaces (Tau, Sandnes, Bryne TWT) into a single unified xarray Dataset saved as `Attributes_Horizons.seisnc`.

3. **`001_seismic_masking.ipynb`** — Applies horizon sculpting to isolate the Bryne Formation (window: 20 samples above to 150 samples below Bryne horizon). Extracts feature vectors within the masked region.

4. **`001_SOM_seis-sculpt_BryneFm_0307.ipynb`** — Main ML workflow: loads masked volume, selects 5 features (Envelope, Phase, Instantaneous Frequency, Energy, Quadrature), normalizes with MinMaxScaler, trains a 5×5 MiniSOM on ~1.7M voxels, maps each voxel to its Best Matching Unit, exports classification cube as SEG-Y and horizon maps for Petrel.

**Data coordinates:** 3D grid indexed by `iline` × `xline` × `samples` (time). The well tie point is at iline=4272, xline=10486.

**Output:** 25-class SOM classification cube exported as SEG-Y for re-import into Petrel.
