# cad_adapter.py
from pathlib import Path
import math
import json
from typing import Dict, List, Tuple, Optional

import ezdxf  # pip install ezdxf
import numpy as np

# If your model expects these columns, list them here (or load from JSON)
DEFAULT_FEATURE_ORDER = ["rotor_diameter_m", "hub_height_m"]

# Optional: map DXF INSUNITS -> meters (tweak if your files use mm/ft/in)
# https://ezdxf.readthedocs.io/en/stable/dxfinternals/units.html
INSUNITS_TO_M = {
    0: 1.0,   # Unitless
    1: 0.0254,  # Inches
    2: 0.3048,  # Feet
    3: 1609.344,  # Miles
    4: 0.001,  # Millimeters
    5: 0.01,   # Centimeters
    6: 1.0,    # Meters
}

# Adjust to your layer naming. Add your real ones for best results.
LAYER_HINTS = {
    "blade": ["BLADE", "ROTOR"],
    "tower": ["TOWER", "MAST", "STRUCT"],
}

def _layer_has(layer: str, hints: List[str]) -> bool:
    L = (layer or "").lower()
    return any(h.lower() in L for h in hints)

def _pts_from_entity(e) -> List[Tuple[float, float]]:
    t = e.dxftype()
    pts: List[Tuple[float, float]] = []
    if t == "LINE":
        pts += [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
    elif t == "LWPOLYLINE":
        for p in e.get_points("xy"):
            pts.append((p[0], p[1]))
    elif t == "CIRCLE":
        cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
        for k in range(36):
            ang = 2 * math.pi * k / 36
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    elif t == "ARC":
        cx, cy, r = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
        a0, a1 = math.radians(e.dxf.start_angle), math.radians(e.dxf.end_angle)
        if a1 < a0:
            a1 += 2 * math.pi
        steps = 36
        for k in range(steps + 1):
            ang = a0 + (a1 - a0) * k / steps
            pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts

def _estimate_units_m(doc) -> float:
    try:
        ins = int(doc.header.get("$INSUNITS", 0))
        return INSUNITS_TO_M.get(ins, 1.0)
    except Exception:
        return 1.0

def _estimate_hub_and_radius(blade_pts: List[Tuple[float, float]]) -> Tuple[Optional[Tuple[float,float]], Optional[float]]:
    if not blade_pts:
        return None, None
    cx = sum(x for x, _ in blade_pts) / len(blade_pts)
    cy = sum(y for _, y in blade_pts) / len(blade_pts)
    dists = sorted(math.hypot(x - cx, y - cy) for x, y in blade_pts)
    if not dists:
        return (cx, cy), None
    top = dists[int(len(dists) * 0.95):] or dists[-3:]
    r = sum(top) / len(top)
    return (cx, cy), r

def _ground_from_tower(tower_pts: List[Tuple[float, float]]) -> float:
    return min((y for _, y in tower_pts), default=0.0)

def extract_turbine_dims_from_dxf(path: Path) -> Dict[str, float]:
    """Return metrics in METERS: rotor_diameter_m, hub_height_m, blade_length_m."""
    doc = ezdxf.readfile(str(path))
    scale = _estimate_units_m(doc)  # convert file units -> meters
    msp = doc.modelspace()

    blade_pts: List[Tuple[float, float]] = []
    tower_pts: List[Tuple[float, float]] = []

    for e in msp:
        t = e.dxftype()
        if t not in ("LINE", "LWPOLYLINE", "CIRCLE", "ARC"):
            continue
        layer = e.dxf.layer or ""
        pts = _pts_from_entity(e)
        if not pts:
            continue

        if _layer_has(layer, LAYER_HINTS["blade"]):
            blade_pts += pts
        elif _layer_has(layer, LAYER_HINTS["tower"]):
            tower_pts += pts
        else:
            # fallback heuristic: very large circle/arc → likely rotor
            if t in ("CIRCLE", "ARC"):
                r = getattr(e.dxf, "radius", 0.0)
                if r and (r * scale) > 5.0:  # 5 m threshold — tune as needed
                    blade_pts += pts

    # scale all points to meters
    blade_pts = [(x * scale, y * scale) for x, y in blade_pts]
    tower_pts = [(x * scale, y * scale) for x, y in tower_pts]

    hub, r = _estimate_hub_and_radius(blade_pts)
    ground_y = _ground_from_tower(tower_pts)

    rotor_diameter_m = 2 * r if r else None
    hub_height_m = (hub[1] - ground_y) if hub else None
    blade_length_m = r if r else None

    return {
        "rotor_diameter_m": float(rotor_diameter_m) if rotor_diameter_m else None,
        "hub_height_m": float(hub_height_m) if hub_height_m else None,
        "blade_length_m": float(blade_length_m) if blade_length_m else None,
    }

def to_model_row(parsed_dims: Dict[str, float], feature_order_path: Optional[str] = None):
    order = DEFAULT_FEATURE_ORDER
    if feature_order_path:
        order = json.load(open(feature_order_path, "r", encoding="utf-8"))
    row = [parsed_dims.get(col, np.nan) for col in order]
    return order, np.array(row, dtype=float).reshape(1, -1)
