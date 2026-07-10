"""Generate a procedural educational long-bone anatomy prototype as GLB.

Usage:
    pip install numpy trimesh
    python generate_bone_anatomy.py
"""
from pathlib import Path
import math
import numpy as np
import trimesh

OUT = Path(__file__).with_name("Bone_Anatomy_Prototype.glb")


def mat(rgba, roughness=0.75):
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=np.array(rgba, dtype=np.uint8),
        metallicFactor=0.0,
        roughnessFactor=roughness,
    )


BONE = mat([224, 211, 181, 255], 0.82)
INNER = mat([194, 159, 112, 255], 0.80)
MARROW = mat([225, 171, 42, 255], 0.63)
SPONGY = mat([139, 48, 36, 255], 0.88)
CARTILAGE = mat([118, 151, 169, 255], 0.42)
ARTERY = mat([160, 33, 35, 255], 0.38)
VEIN = mat([48, 69, 125, 255], 0.42)


def paint(mesh, material):
    mesh.visual = trimesh.visual.TextureVisuals(material=material)
    return mesh


def ellipsoid(scale, center):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mesh.apply_scale(scale)
    mesh.apply_translation(center)
    return mesh


def cylinder_between(p0, p1, radius, sections=16):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vector = p1 - p0
    length = np.linalg.norm(vector)
    mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)
    mesh.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], vector / length))
    mesh.apply_translation((p0 + p1) / 2)
    return mesh


def cutaway_shell(z_values, outer_r, inner_r, n_theta=72):
    theta = np.linspace(math.radians(40), math.radians(320), n_theta)
    vertices, faces, rings = [], [], []
    for z, ro, ri in zip(z_values, outer_r, inner_r):
        outer, inner = [], []
        for t in theta:
            outer.append(len(vertices))
            vertices.append([ro * math.cos(t), 0.82 * ro * math.sin(t), z])
        for t in theta:
            inner.append(len(vertices))
            vertices.append([ri * math.cos(t), 0.82 * ri * math.sin(t), z])
        rings.append((outer, inner))

    for index in range(len(rings) - 1):
        o0, i0 = rings[index]
        o1, i1 = rings[index + 1]
        for j in range(n_theta - 1):
            faces.extend([
                [o0[j], o1[j], o1[j + 1]], [o0[j], o1[j + 1], o0[j + 1]],
                [i0[j], i1[j + 1], i1[j]], [i0[j], i0[j + 1], i1[j + 1]],
            ])
        for j in (0, n_theta - 1):
            faces.extend([[o0[j], i0[j], i1[j]], [o0[j], i1[j], o1[j]]])

    for ring_index, (outer, inner) in enumerate((rings[0], rings[-1])):
        for j in range(n_theta - 1):
            f1 = [outer[j], inner[j + 1], inner[j]]
            f2 = [outer[j], outer[j + 1], inner[j + 1]]
            if ring_index == 0:
                f1.reverse(); f2.reverse()
            faces.extend([f1, f2])

    return trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=True)


def build_scene():
    scene = trimesh.Scene()

    z = np.linspace(-4.6, 4.5, 34)
    outer = 0.65 + 0.13 * (np.abs(z) / 4.6) ** 1.8
    outer += 0.08 * np.exp(-((z - 3.9) / 0.9) ** 2)
    inner = np.maximum(outer - 0.18, 0.34)
    scene.add_geometry(paint(cutaway_shell(z, outer, inner), BONE), node_name="Compact_Bone_Cutaway")

    parts = {
        "Distal_Condyle_Medial": ([0.78, 0.72, 0.88], [-0.52, 0, -5.15]),
        "Distal_Condyle_Lateral": ([0.78, 0.72, 0.88], [0.52, 0, -5.15]),
        "Proximal_Neck": ([0.78, 0.62, 1.05], [0.20, 0, 4.75]),
        "Femoral_Head": ([1.08, 1.00, 1.05], [-0.72, 0, 5.35]),
        "Greater_Trochanter": ([0.65, 0.60, 0.95], [0.78, 0, 4.72]),
    }
    for name, (scale, center) in parts.items():
        scene.add_geometry(paint(ellipsoid(scale, center), BONE), node_name=name)

    cartilage = {
        "Articular_Cartilage_Proximal": ([1.105, 1.025, 1.075], [-0.72, 0, 5.39]),
        "Articular_Cartilage_Distal_Medial": ([0.80, 0.74, 0.28], [-0.52, 0, -5.73]),
        "Articular_Cartilage_Distal_Lateral": ([0.80, 0.74, 0.28], [0.52, 0, -5.73]),
    }
    for name, (scale, center) in cartilage.items():
        scene.add_geometry(paint(ellipsoid(scale, center), CARTILAGE), node_name=name)

    marrow = trimesh.creation.cylinder(radius=0.36, height=7.8, sections=40)
    marrow.apply_translation([0, 0, -0.15])
    scene.add_geometry(paint(marrow, MARROW), node_name="Medullary_Canal_Yellow_Marrow")

    rng = np.random.default_rng(7)
    points = []
    for _ in range(42):
        while True:
            p = rng.uniform([-1.25, -0.72, 4.25], [0.95, 0.72, 6.0])
            q = np.array([(p[0] + 0.2) / 1.25, p[1] / 0.72, (p[2] - 5.15) / 1.15])
            if np.dot(q, q) < 1.0:
                points.append(p); break
    cloud = np.asarray(points)
    for i in range(62):
        a = points[rng.integers(0, len(points))]
        nearest = np.argsort(np.linalg.norm(cloud - a, axis=1))[1:7]
        b = points[int(rng.choice(nearest))]
        scene.add_geometry(paint(cylinder_between(a, b, 0.035, 10), SPONGY), node_name=f"Trabecula_{i:02d}")

    vessels = [
        ("Nutrient_Artery", [0.18, -0.73, -0.9], [0.12, -0.08, -0.15], 0.055, ARTERY),
        ("Artery_Branch_1", [0.12, -0.08, -0.15], [-0.16, 0, 0.55], 0.035, ARTERY),
        ("Artery_Branch_2", [0.12, -0.08, -0.15], [0.30, 0.02, 0.65], 0.032, ARTERY),
        ("Nutrient_Vein", [0.30, -0.70, -0.95], [0.27, -0.02, 0.05], 0.042, VEIN),
    ]
    for name, p0, p1, radius, material in vessels:
        scene.add_geometry(paint(cylinder_between(p0, p1, radius), material), node_name=name)

    center = np.array([3.0, 0.0, 1.0])
    block = trimesh.creation.box(extents=[2.2, 2.2, 2.4])
    block.apply_translation(center)
    scene.add_geometry(paint(block, INNER), node_name="Compact_Bone_Microstructure_Block")

    for ix in (-0.65, 0, 0.65):
        for iy in (-0.65, 0, 0.65):
            c = center + np.array([ix, iy, 1.22])
            for radius in (0.27, 0.20, 0.13):
                ring = trimesh.creation.annulus(
                    r_min=radius - 0.018, r_max=radius + 0.018,
                    height=0.025, sections=32,
                )
                ring.apply_translation(c)
                scene.add_geometry(paint(ring, BONE), node_name="Concentric_Lamella")
            canal = trimesh.creation.cylinder(radius=0.055, height=0.34, sections=20)
            canal.apply_translation(c + np.array([0, 0, -0.14]))
            scene.add_geometry(paint(canal, ARTERY), node_name="Haversian_Canal")

    return scene


if __name__ == "__main__":
    scene = build_scene()
    scene.export(OUT)
    print(f"Created {OUT} with {len(scene.geometry)} geometries")
