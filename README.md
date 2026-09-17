# GEOSTIX Metashape Plugin

Two Agisoft Metashape Python plugins for processing photogrammetry surveys captured with a
[GEOSTIX](https://geostix.com/) rig: importing images geotagged from the GEOSTIX GNSS log, and
aligning/analysing the resulting chunk against that GNSS reference.

Both plugins register themselves under a **GEOSTIX** menu in Metashape when run.

## Plugins

### 1. Import images and GNSS (`plugin_import_geostix.py`)

Menu item: **GEOSTIX → Import images and GNSS**

Creates a new chunk from a folder of photos and assigns each camera a reference location/accuracy
taken from a GEOSTIX GNSS log, matching images to log rows by timestamp rather than relying on
manual pairing.

What it does:
- Reads capture time from each photo's EXIF (`DateTimeOriginal` + sub-seconds), converted to GPS
  time of week.
- Parses the GEOSTIX log CSV (time of week, fix status, lat/lon/height, horizontal/vertical
  accuracy, orientation, etc.).
- Because the camera clock and GNSS receiver clock aren't synchronized, it estimates the time
  offset between the two series (coarse window search, refined with `scipy.optimize.minimize`)
  and shifts the image timestamps to match the log.
- Matches each image to its nearest log entry (`pandas.merge_asof`, with a tolerance) and creates
  a new chunk with those photos, setting `camera.reference.location` /
  `location_accuracy` from the matched GNSS fix.
- Lets you pick a target CRS from a shortlist (WGS 84, CGRS 94) or any other CRS via Metashape's
  own coordinate system picker.

Dialog fields: image directory, GEOSTIX log file, coordinate system.

### 2. Align and analyse (`plugin_align_and_analyse.py`)

Menu item: **GEOSTIX → Align and analyse**

Runs photo alignment on the active chunk twice — once unconstrained, once optimized against the
GNSS reference — and reports how well the two agree.

What it does:
1. Disables GNSS references, runs `matchPhotos`/`alignCameras` at the chosen accuracy
   (with optional generic/reference preselection).
2. Applies a fixed camera→antenna lever arm offset, then re-enables the GNSS references and
   updates the chunk transform ("initial" alignment).
3. Unlocks the lever arm and runs `optimizeCameras` to let it refine freely ("optimized"
   alignment).
4. For both stages, computes per-camera error statistics: estimated vs. GNSS-reference position
   in a local tangent frame, converted to both metric and accuracy-weighted ("unit") errors, plus
   an overall `m0` reprojection-style error metric.
5. Opens a results dialog with an **Initial** / **Optimized** tab, each showing:
   - the `m0` summary label,
   - discrete-colored polar histograms of the metric error vectors in the XY, XZ and YZ planes
     (angle = error direction, radius = error magnitude, color = point count per bin), and
   - a regular histogram of unit errors.

Dialog fields: accuracy preset, lever arm (X/Y/Z), generic/reference preselection toggles.

## Project layout

| File | Purpose |
|---|---|
| `plugin_import_geostix.py` | Import plugin logic + menu registration |
| `plugin_align_and_analyse.py` | Align/analyse plugin logic + menu registration |
| `import_geostix.ui` | Qt Designer dialog for the import plugin |
| `align.ui` | Qt Designer dialog for the align/analyse plugin's input parameters |
| `plot.ui` | Qt Designer dialog for the align/analyse results (tabs + plots) |
| `mplwidget.py` | Reusable `QWidget` embedding a Matplotlib canvas + navigation toolbar, used by `plot.ui` |

## Requirements

- Agisoft Metashape Professional (Python 3, PySide2, and the `Metashape` module it ships with).
- A `modules.pip_auto_install` helper module available on Metashape's Python path (each plugin
  calls `pip_install(...)` on load to ensure its third-party dependencies — `numpy`, `matplotlib`,
  `pandas`, `scipy`, `pillow` — are installed into Metashape's Python environment).

## Installation

Copy the `.py` files together with their matching `.ui` files and `mplwidget.py` into Metashape's
scripts folder (or wherever your `modules.pip_auto_install` helper is discoverable from), then run
each script once from Metashape's Python console / **Tools → Run Script...** to register its menu
item. Re-running a script first removes its previous menu item, so it's safe to reload during
development.
