# Changelog

## 0.2.0
- **Timing reporting:** `run_tomofast_blobs`, `run_source_detection`, and `run_dipole_fitting` now record wall-clock time and print total stage duration on completion (`Tool 1/2/3 total: X.XX min`).
- **Bash timing wrap:** `_run_live()` gains a `bash_time=False` parameter; when `True`, wraps the command as `{ time CMD; } 2>&1` so shell timing is captured in logs. Used automatically by all mpirun calls.
- **Progress tracking:** `run_source_detection` displays a single-line overwriting progress bar (`\rProgress: done/total | blob grain mesh | cost=...`) via the new `_print_progress()` helper.
- **`extract_blob_data_costs(blob_output_dir)`** (new): Parses `Blob_XXX/QDM_log_Blob_XXX.txt` for the last "data cost (new)" value per blob; returns a `DataFrame(blob, final_data_cost)`.
- **`compute_field_data_cost(obs_path, calc_path)`** (new): Reads two Tomofast `.obs` files and returns the relative L2 misfit `‖bz_pred − bz_obs‖₂ / ‖bz_obs‖₂`.
- Both new analysis functions are exported from `tomofast_x_mu` top-level namespace.

## 0.1.0
- Initial packaging of the Tomofast-x-µ QDM preprocessing + analysis workflow.
- Added robust handling of QDM `.mat` standoff (`h`) units via `h_units={m,um,auto}`.
