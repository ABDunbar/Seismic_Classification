# Code Review: Seismic Classification Notebooks

## Bugs

**`001_load_Petrel_seismic_attribute_segys.ipynb`, Cell 15**
References `cube` which is never defined in this notebook — will raise `NameError`. This visualization cell either needs `cube` loaded, or should be moved to a notebook that has it.

**`001_seismic_masking.ipynb` / `001_SOM_seis-sculpt_BryneFm_0307.ipynb`, Cell 47**
```python
# Bug: variable name mismatch
sandnes_som = cube.SOM_labels.interp(...)      # ← doesn't exist
# Was added as:
cube["SOM_Sandnes5x5"] = SOM.som_labels2       # ← correct name
```

---

## Correctness / Logic

**Masking arithmetic is unsound (`001_seismic_masking.ipynb`, Cell 8)**
```python
mask_below = cube.where(cube.samples < cube.Bryne_TWT + 150)
mask_above = cube.where(cube.samples > cube.Bryne_TWT - 20)
masked_data = 0.5 * (mask_below + mask_above)   # ← WRONG
```
`xr.where()` replaces out-of-range values with `NaN`, so adding two masked datasets and dividing by 2 gives incorrect values: voxels inside the window get the correct value from only one mask, but the averaging by 0.5 halves the values. The semantically correct approach is boolean intersection:
```python
mask = (cube.samples > cube.Bryne_TWT - 20) & (cube.samples < cube.Bryne_TWT + 150)
masked_data = cube.where(mask)
```

**Unused `masks` list**
```python
masks = [mask_above, mask_below]   # defined but never used
```

**BMU label assignment is slow (Cell 37)**
```python
bmu_labels = np.array([som.winner(x) for x in X])  # pure Python loop over 1.7M samples
```
MiniSom supports batch winner assignment via vectorized distance computation:
```python
dists = som._distance_from_weights(X)
bmu_labels = np.array([np.unravel_index(d.argmin(), (SOM_X_AXIS_NODES, SOM_Y_AXIS_NODES)) for d in dists])
```
The loop as written will be very slow on 1.7M points.

---

## Extreme Code Duplication

**`001_load_Petrel_seismic_attribute_segys.ipynb`** — 13 nearly identical cells, one per SEG-Y file. This should be a loop:
```python
attributes = {
    "amplitude.segy": "Amplitudes.nc",
    "3D_edge_enh.segy": "Edge_3D_enhance.nc",
    "AppPol_w33.segy": "Apparent_polarity_w33.nc",
    # ... etc
}

for segy_file, nc_file in attributes.items():
    seis_path = pathlib.Path(f"Attributes/{segy_file}")
    print("3D", seis_path, seis_path.exists())
    ds = xr.open_dataset(
        seis_path,
        dim_byte_fields={"iline": 5, "xline": 21},
        extra_byte_fields={"cdp_x": 73, "cdp_y": 77},
    )
    ds = ds.set_coords(("cdp_x", "cdp_y"))
    ds.to_netcdf(f"Attributes/{nc_file}")
```

**`001_merge_attribute_volumes.ipynb`** — 12 identical cells loading `.nc` files. Should be:
```python
attrs = {"apppol": "Apparent_polarity_w33.nc", "chaos": "Chaos.nc", ...}
for var_name, nc_file in attrs.items():
    volume[var_name] = xr.open_dataset(f"Attributes/{nc_file}").data
```

**Repeated plotting boilerplate** — The same 10-line horizon overlay plot block is copy-pasted ~8 times across the masking/SOM notebooks. Extract it:
```python
def plot_xline_section(data, xline, samples_slice, cmap, cube):
    opt = dict(x="iline", y="samples", add_colorbar=True,
               interpolation="spline16", robust=True, yincrease=False, cmap=cmap)
    f, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    data.sel(xline=xline, samples=samples_slice).plot.imshow(ax=ax, **opt)
    for horizon, color in [(cube.Tau_TWT, "b"), (cube.Sandnes_TWT, "orange"), (cube.Bryne_TWT, "r")]:
        ax.plot(horizon.sel(xline=xline).iline, horizon.sel(xline=xline), color=color)
    ax.invert_xaxis()
    return ax
```

**Notebooks 3 and 4 are duplicates** — `001_seismic_masking.ipynb` and `001_SOM_seis-sculpt_BryneFm_0307.ipynb` appear to be identical. One should be deleted or they should be differentiated.

---

## Efficiency

**`001_merge_attribute_volumes.ipynb`** — Each `.nc` file is opened, one variable extracted, then the file handle is left open (no `with` block, no `.close()`). xarray keeps lazy handles open, but explicit closing or using context managers prevents handle leaks across a long session:
```python
with xr.open_dataset("Attributes/Chaos.nc") as ds:
    volume["chaos"] = ds.data.load()  # .load() materializes before handle closes
```

**`to_dataframe()` on a 20M-row Dataset** (Cell 22) materializes the entire sparse masked volume into RAM before `dropna()`. Better to work in xarray until the last moment:
```python
# Instead of:
df = seis_sculpt.to_dataframe().dropna()
X_orig = df[feature_columns].to_numpy()

# Do:
valid_mask = seis_sculpt[feature_columns[0]].notnull()
X_orig = np.stack([seis_sculpt[c].values[valid_mask.values] for c in feature_columns], axis=1)
```
This avoids a ~1GB+ intermediate DataFrame.

**SOM training iteration count** — 5,000 iterations over 1.7M samples with random sampling means only a tiny fraction of data is seen per epoch. Consider `som.train_batch(X, n_epochs)` which guarantees full data passes, or increase iterations proportionally.

---

## Minor / Best Practices

- `iline = 4272` / `xline = 10486` are redefined multiple times across cells and notebooks as magic numbers. Define them once at the top as constants: `WELL_ILINE`, `WELL_XLINE`.
- `RANDOM_SEED = 123` is defined but could also seed `numpy` (`np.random.seed(123)`) for the `np.random.choice` sampling to ensure full reproducibility.
- `from sklearn.mixture import GaussianMixture` is imported but never used.
- `import bruges as bg` / `import platform` appear in every notebook but are unused.
- The `sample_cube_near_well()` function is defined but only used in a commented-out cell — delete it or uncomment the usage.
- The `color_map` visualization (Cell 45) clips to only 3 of 5 features when building an RGB image, silently discarding 2 dimensions. Add a comment or use PCA to project to 3 components properly.
