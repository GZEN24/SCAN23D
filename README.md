# SCAN23D

SCAN23D contains Python source code for RGB-D capture, 3D reconstruction, and orthopedic cast shell generation. The repository is kept source-focused: raw camera frames, generated meshes, reports, posters, and presentation files are intentionally excluded.


## Citation

If you use SCAN23D, its source code, or its methodology in your research, please cite the following peer-reviewed publication:

AlAjmi, A., AlHajeri, N., AlAjmi, R., Kamshad, S., AlFarraj, L., and Zakka El Nashef, G. (2026). "Low-Cost, Open-Source Design and FDM Fabrication of Patient-Specific Orthopedic Forearm Casts from Consumer RGB-D Scanning." *3D Printing and Additive Manufacturing*. OnlineFirst.

**DOI:** https://doi.org/10.1177/23297662261477132

**Source code:** https://github.com/GZEN24/SCAN23D

For reproducibility, please also reference the SCAN23D GitHub repository and the specific commit or release version used.


## Repository layout

- `SCAN23D-CODE/g18_capture.py` - Intel RealSense D435 capture utility.
- `SCAN23D-CODE/g18_reconstruct.py` - RGB-D reconstruction and cast generation pipeline.
- `SCAN23D-CODE/g18_unified_pipeline.py` - unified RGB-D/RGB pipeline with synthetic-data support.
- `SCAN23D-CODE/G18_Visual_Walkthrough_RS.ipynb` - source notebook for the RealSense walkthrough, with outputs stripped.
- `Design Schematics/` - retained design reference material.

## Setup

Use a virtual environment before installing dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy opencv-python open3d trimesh scipy scikit-image manifold3d
```

Optional dependencies:

```powershell
pip install pyrealsense2
pip install rembg onnxruntime
```

## Usage

Run commands from `SCAN23D-CODE`:

```powershell
cd SCAN23D-CODE
python g18_capture.py --save D:\captures\g18_capture
python g18_unified_pipeline.py --synthetic
```

For real capture reconstruction, keep capture data outside the repository or in the ignored local capture paths (`rgb/`, `depth/`, `background.png`, `capture_log.json`, and `intrinsics.json`). Generated output folders such as `g18_output/` are ignored.
