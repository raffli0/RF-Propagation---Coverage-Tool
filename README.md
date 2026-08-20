# RF-Propagation---Coverage-Tool

This project is designed to model and analyze radio frequency (RF) propagation and coverage. The tool provides simulations and visualizations for understanding how RF signals behave in various environments. It can be used for network planning, optimization, and other applications related to wireless communication systems.

## Features

## donwload maps
## https://drive.google.com/file/d/1h6Xxj4x3DZsFAy2Xz-Kb9wWoYMjlTntP/view?usp=drive_link

- **RF Propagation Modeling**: Simulate RF signal propagation in different environments (Free Space, Rain, Gas, Fog, Close-In, Longley-Rice, TIREM, Ray Tracing).
- **Coverage Analysis**: Visualize coverage maps (path loss / elevation heatmaps) and determine signal strength in specific areas.
- **Offline Elevation**: SRTM-derived terrain elevation for map areas, cached locally under `~/.cache/rf_propagation` after the first download.

## Project structure

```
app.py           Streamlit UI entry point (orchestration only)
config.py        Global constants and UI option lists
colormaps.py     Colormap LUTs, value -> RGBA mapping, colorbar rendering
map_utils.py     Folium map construction, overlays and drawn-area handling
heatmap.py       Raster grids, cached path-loss/elevation assets, heatmap rendering
models.py        RF propagation models + model dispatch
elevation.py     Offline SRTM terrain elevation sampling
```

## Installation

### Prerequisites

- Python 3.8 or higher

### Clone the repository

```bash
git clone https://github.com/umutonuryasar/RF-Propagation---Coverage-Tool.git
cd RF-Propagation---Coverage-Tool
```

### Install the required packages

```bash
pip install -r requirements.txt
```

### Usage

- Configure Parameters: Adjust the parameters in the UI.
- Run the Tool: Execute the app.py script to perform the RF propagation simulation.

```bash
streamlit run app.py
```

- Select a propagation model and adjust Frequency.
- Set the transmitter location (Latitude/Longitude) and link Distance.
- Optionally enable the heatmap in the sidebar and draw a polygon on the map to define the coverage contour.
- Click **Calculate Coverage** to compute the path loss and view the coverage circle on the map.

### License

This project is licensed under the MIT License - see the LICENSE file for details.
