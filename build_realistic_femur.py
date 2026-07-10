"""Build a realistic femur GLB from the open BMFToolkit CT mesh dataset.

Source dataset:
  Manish Sreenivasa and Daniel Gonzalez-Alvarado,
  Bone Mesh Female Toolkit (BMFToolkit), zlib license.

The script downloads the original MATLAB mesh dataset, extracts one femur,
normalizes it, adds educational cutaway components, and exports a GLB.
"""
from __future__ import annotations

from pathlib import Path
import io
import urllib.request

import numpy as np
import scipy.io
import trimesh

SOURCE_URL = (
    "https://raw.githubusercontent.com/manishsreenivasa/"
    "BMFToolkit/master/data/model_Original.mat"
)
OUT = Path(__file__).with_name("Realistic_Femur_Anatomy.glb")


def material(rgba, roughness=0.78):
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.asarray(rgba, dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=roughness,
    )


BONE = material([220, 207, 180, 255], 0.86)
MARROW = material([222, 157, 43, 255], 0.66)
CARTILAGE = material([123, 156, 174, 255], 0.40)
ARTERY = material([165, 38, 42, 255], 0.38)
SPONGY = material([145, 57, 43, 255], 0.90)


def paint(mesh: trimesh.Trimesh, mat):
    mesh.visual = trimesh.visual.TextureVisuals(material=mat)
    return mesh


def _field(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    return None


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    arr = np.asarray(value)
    if arr.dtype.kind in "US":
        return "".join(arr.ravel().tolist())
    if arr.size == 1:
        return str(arr.item())
    return str(value)


def _flatten_models(value):
    if isinstance(value, np.ndarray):
        for item in value.ravel():
            yield item
    elif isinstance(value, (list, tuple)):
        yield from value
    else:
        yield value


def load_femur() -> trimesh.Trimesh:
    print("Downloading BMFToolkit CT mesh dataset...")
    with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
        raw = response.read()
    data = scipy.io.loadmat(
        io.BytesIO(raw),
        squeeze_me=True,
        struct_as_record=False,
    )
    models = data.get("model_Original")
    if models is None:
        raise RuntimeError("model_Original not found in source dataset")

    candidates = []
    for model in _flatten_models(models):
        name = _text(_field(model, "BoneName")).strip()
        vertices = _field(model, "vertices_global")
        faces = _field(model, "faces")
        if vertices is None or faces is None:
            continue
        if "femur" in name.lower():
            candidates.append((name, np.asarray(vertices, float), np.asarray(faces, int)))

    if not candidates:
        # Dataset documentation indicates entries 4 and 5 are femora.
        flat = list(_flatten_models(models))
        for index in (3, 4):
            model = flat[index]
            candidates.append((
                _text(_field(model, "BoneName")) or f"Femur_{index}",
                np.asarray(_field(model, "vertices_global"), float),
                np.asarray(_field(model, "faces"), int),
            ))

    name, vertices, faces = candidates[0]
    if faces.min() == 1:
        faces = faces - 1
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    print(f"Using {name}: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # Center, align longest principal axis to +Z, and scale to 44 cm.
    mesh.apply_translation(-mesh.centroid)
    transform = mesh.principal_inertia_transform
    mesh.apply_transform(transform)
    extents = mesh.extents
    long_axis = int(np.argmax(extents))
    if long_axis != 2:
        axes = [0, 1, 2]
        axes[long_axis], axes[2] = axes[2], axes[long_axis]
        matrix = np.eye(4)
        matrix[:3, :3] = np.eye(3)[:, axes]
        mesh.apply_transform(matrix)
    mesh.apply_scale(0.44 / mesh.extents[2])
    mesh.apply_translation(-mesh.centroid)
    return mesh


def cylinder_between(p0, p1, radius, sections=24):
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    vector = p1 - p0
    length = np.linalg.norm(vector)
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], vector / length))
    mesh.apply_translation((p0 + p1) / 2)
    return mesh


def build_scene() -> trimesh.Scene:
    femur = load_femur()
    scene = trimesh.Scene()
    scene.add_geometry(paint(femur, BONE), node_name="CT_Derived_Femur")

    bounds = femur.bounds
    zmin, zmax = bounds[0, 2], bounds[1, 2]
    height = zmax - zmin
    shaft_radius = max(femur.extents[0], femur.extents[1]) * 0.08

    # Educational internal structures placed inside the diaphysis.
    marrow = trimesh.creation.cylinder(
        radius=max(shaft_radius * 0.55, 0.006),
        height=height * 0.55,
        sections=40,
    )
    marrow.apply_translation([0, 0, zmin + height * 0.48])
    scene.add_geometry(paint(marrow, MARROW), node_name="Yellow_Bone_Marrow")

    artery = cylinder_between(
        [shaft_radius * 1.5, -shaft_radius * 1.8, zmin + height * 0.38],
        [0, 0, zmin + height * 0.58],
        max(shaft_radius * 0.08, 0.0012),
    )
    scene.add_geometry(paint(artery, ARTERY), node_name="Nutrient_Artery")

    # Separate compact-bone teaching sample with osteons.
    sample_center = np.array([femur.extents[0] * 1.8, 0, 0])
    sample = trimesh.creation.cylinder(radius=0.035, height=0.055, sections=64)
    sample.apply_translation(sample_center)
    scene.add_geometry(paint(sample, BONE), node_name="Compact_Bone_Sample")

    for x, y in [(-0.014, -0.014), (0.014, -0.014), (-0.014, 0.014), (0.014, 0.014), (0, 0)]:
        center = sample_center + np.array([x, y, 0.028])
        for radius in (0.008, 0.0058, 0.0037):
            ring = trimesh.creation.annulus(
                r_min=radius - 0.00055,
                r_max=radius + 0.00055,
                height=0.001,
                sections=36,
            )
            ring.apply_translation(center)
            scene.add_geometry(paint(ring, BONE), node_name="Concentric_Lamella")
        canal = trimesh.creation.cylinder(radius=0.0014, height=0.012, sections=18)
        canal.apply_translation(center + np.array([0, 0, -0.005]))
        scene.add_geometry(paint(canal, ARTERY), node_name="Haversian_Canal")

    return scene


if __name__ == "__main__":
    result = build_scene()
    result.export(OUT)
    print(f"Created {OUT.name} with {len(result.geometry)} geometries")
