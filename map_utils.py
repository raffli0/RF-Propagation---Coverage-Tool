"""Folium map construction and overlay helpers (coverage, terrain, drawing)."""

import folium
import numpy as np
from folium.plugins import Draw
from folium.raster_layers import ImageOverlay


def create_map(location, map_style="OpenStreetMap", add_draw=False, max_zoom=19):
    m = folium.Map(location=location, zoom_start=13, control_scale=True, max_zoom=max_zoom)

    if map_style == "Topografi":
        folium.TileLayer(
            tiles="https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
            attr="Map data: © OpenStreetMap contributors, SRTM | map style: © OpenTopoMap",
            name="Topografi",
            overlay=False,
            opacity=0.98,
        ).add_to(m)
        m.options["prefer_canvas"] = True
        m.options["attributionControl"] = True
        m.get_root().script.add_child(folium.Element("""
            <style>
                .leaflet-container {
                    background: #cbd7d0;
                }
                .leaflet-control-attribution {
                    font-size: 9px;
                    background: rgba(255,255,255,0.7);
                    padding: 2px 6px;
                }
                .leaflet-label {
                    font-size: 9px !important;
                    font-weight: 600 !important;
                    color: rgba(40, 40, 40, 0.8) !important;
                    background: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                    text-shadow: 0 0 2px rgba(255,255,255,0.9);
                }
            </style>
        """))
    elif map_style == "Satelit":
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/Tile/{z}/{y}/{x}",
            attr="Tiles © Esri",
            name="Satelit",
            overlay=False,
        ).add_to(m)
    else:
        folium.TileLayer(
            tiles="OpenStreetMap",
            attr="© OpenStreetMap contributors",
            name="OpenStreetMap",
            overlay=False,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    if add_draw:
        Draw(
            export=False,
            position="topleft",
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "rectangle": False,
                "marker": False,
                "circlemarker": False,
            },
            edit_options={"edit": False, "remove": False},
        ).add_to(m)
    return m


def update_map_with_coverage(map_obj, location, path_loss, distance):
    folium.Circle(
        location=location,
        radius=distance * 1000,
        color='yellow',
        weight=2,
        fill=True,
        fill_color='yellow',
        fill_opacity=0.5,
        popup=f"Path Loss: {path_loss:.2f} dB"
    ).add_to(map_obj)
    return map_obj


def remove_folium_circle(map_obj):
    """Remove the coverage Circle added by update_map_with_coverage from the map."""
    for child_id in list(map_obj._children.keys()):
        child = map_obj._children.get(child_id)
        if isinstance(child, folium.Circle):
            del map_obj._children[child_id]
    return map_obj


def add_coverage_raster(map_obj, rgba, bounds, opacity):
    """Add the path loss grid as a raster ImageOverlay."""
    ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=opacity,
        name="Coverage Heatmap",
        pixelated=False,
        interactive=False,
        control=False,
    ).add_to(map_obj)
    return map_obj


def add_colorbar_overlay(map_obj, cbar_img, grid_bounds, position="bottomright"):
    """Add the colorbar PNG as an ImageOverlay anchored to a corner of the grid."""
    lat_min, lon_min = grid_bounds[0]
    lat_max, lon_max = grid_bounds[1]
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    cw = lon_span * 0.34
    ch = cw * (cbar_img.size[1] / cbar_img.size[0])
    mx = lon_span * 0.015
    my = lat_span * 0.02
    if position == "bottomright":
        corner = (lon_max - cw - mx, lat_min + my)
    elif position == "bottomleft":
        corner = (lon_min + mx, lat_min + my)
    elif position == "topright":
        corner = (lon_max - cw - mx, lat_max - ch - my)
    else:
        corner = (lon_min + mx, lat_max - ch - my)
    ImageOverlay(
        image=np.asarray(cbar_img),
        bounds=[[corner[1], corner[0]], [corner[1] + ch, corner[0] + cw]],
        opacity=1.0,
        name="Colorbar",
        interactive=False,
        control=False,
    ).add_to(map_obj)
    return map_obj


def add_selection_polygon(map_obj, ring):
    """Re-add the user's drawn area so it survives map remounts."""
    folium.Polygon(
        locations=ring,
        color="#0d47a1",
        weight=2,
        fill=True,
        fill_color="#42a5f5",
        fill_opacity=0.12,
        popup="Coverage area selection",
    ).add_to(map_obj)
    return map_obj


def add_tx_marker(map_obj, location, label="TX"):
    """Drop a small transmitter marker at the site location."""
    folium.CircleMarker(
        location=location,
        radius=6,
        color="#ffd166",
        weight=2,
        fill=True,
        fill_color="#ffd166",
        fill_opacity=1.0,
        popup=label,
        tooltip=label,
    ).add_to(map_obj)
    return map_obj


def add_location_readout(map_obj, location, mode_text=""):
    """Pin a small coordinate/readout chip to the bottom-left of the map."""
    lat, lon = location
    text = f"TX: {lat:.5f}, {lon:.5f}"
    if mode_text:
        text += f" &nbsp;|&nbsp; {mode_text}"
    html = f"""
        <div class="rf-map-readout">{text}</div>
    """
    map_obj.get_root().html.add_child(folium.Element(html))
    return map_obj


def extract_drawn_polygon(map_data):
    """Extract the closed [lat, lon] ring of the last drawn shape.

    Returns a list of [lat, lon] vertices, or None when nothing was drawn.
    """
    if not map_data:
        return None
    drawing = map_data.get("last_active_drawing")
    if not drawing:
        return None
    geometry = drawing.get("geometry") or {}
    coords = geometry.get("coordinates") or []
    if not coords or not coords[0]:
        return None
    ring = coords[0]
    if len(ring) < 3:
        return None
    return [[pt[1], pt[0]] for pt in ring]



