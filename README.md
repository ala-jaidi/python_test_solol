# Podiatry Foot Measurement API

FastAPI backend for measuring foot dimensions from photos using **Segment Anything (SAM)** segmentation and **ArUco** marker calibration, with **DXF** export for orthopedic insole manufacturing.

## Features

- **Segmentation** — Automatic foot detection with SAM (vit_b)
- **Calibration** — ArUco L-board markers for real-world scale
- **Measurements** — Length, width, toe angle from top and side views
- **DXF Export** — Professional contour export for CAD/CAM workflows
- **REST API** — FastAPI endpoints for Flutter mobile app integration

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── api.py                  # FastAPI routes and server
│   ├── mobile_sam_podiatry.py  # SAM segmentation + measurement pipeline
│   ├── utils.py                # Image processing utilities
│   └── dxf_export.py           # DXF file generation
├── models/                     # SAM weights (auto-downloaded, gitignored)
├── render.yaml                 # Render deployment blueprint
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

> The SAM model (~375 MB) is downloaded automatically on first startup.

## Run Locally

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

The API docs are available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/measure/top/` | Top view — width + toe angle |
| `POST` | `/measure/side/` | Side view — heel-to-toe length |
| `POST` | `/measure/complete/` | Final measurements + shoe size |

### POST /measure/top/ and /measure/side/

Multipart form data:
- **image** (file) — Photo with ArUco L-board visible
- **foot_side** (string) — `"left"` or `"right"`

### POST /measure/complete/

JSON body:
```json
{
  "left_foot":  { "width_cm": 9.5, "length_cm": 26.0, "toe_angle_deg": 15.0 },
  "right_foot": { "width_cm": 9.3, "length_cm": 25.8, "toe_angle_deg": 14.5 }
}
```

## Deploy on Render

This project includes a `render.yaml` blueprint for one-click deployment:

1. Push this repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com) → **New Blueprint**
3. Select the repository and branch `main`
4. Render will auto-detect `render.yaml` and configure the service

> **Note:** The Render Starter plan ($7/mo) is recommended. A persistent disk stores the SAM model so it only downloads once.

## Runtime Directories (gitignored)

- `models/` — SAM checkpoint weights
- `uploads/` — Uploaded images (temporary)
- `output/` — Debug images and DXF exports
