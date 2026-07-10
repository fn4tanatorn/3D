from pathlib import Path
import math, numpy as np, trimesh

OUT=Path(__file__).with_name("Bone_Anatomy_Prototype.glb")

def mat(rgba, roughness=0.75):
    return trimesh.visual.material.PBRMaterial(baseColorFactor=np.array(rgba,dtype=np.uint8),metallicFactor=0.0,roughnessFactor=roughness)
BONE=mat([215,199,166,255],.84); CORTICAL=mat([232,220,191,255],.82); MARROW=mat([224,161,38,255],.62)
SPONGY=mat([135,46,35,255],.88); CARTILAGE=mat([112,146,169,255],.38); ARTERY=mat([170,35,38,255],.34); VEIN=mat([48,73,132,255],.40)

def paint(m,material): m.visual=trimesh.visual.TextureVisuals(material=material); return m
def ellipsoid(scale,center,sub=4):
    m=trimesh.creation.icosphere(subdivisions=sub,radius=1.0); m.apply_scale(scale); m.apply_translation(center); return m
def cyl_between(p0,p1,r,sections=28):
    p0,p1=np.asarray(p0,float),np.asarray(p1,float); v=p1-p0; L=np.linalg.norm(v)
    m=trimesh.creation.cylinder(radius=r,height=L,sections=sections)
    m.apply_transform(trimesh.geometry.align_vectors([0,0,1],v/L)); m.apply_translation((p0+p1)/2); return m

def curved_cutaway_shaft(nz=70,ntheta=96):
    z=np.linspace(-4.6,4.15,nz); cx=.12*np.sin((z+4.6)/8.75*math.pi)-.04; cy=.03*np.sin((z+4.6)/8.75*2*math.pi)
    rx=.55+.10*(np.abs(z)/4.6)**1.8+.08*np.exp(-((z-3.5)/.9)**2); ry=.46+.08*(np.abs(z)/4.6)**1.6
    thk=.13+.03*np.exp(-(z/2.8)**2); irx,iry=rx-thk,ry-thk*.95
    th=np.linspace(math.radians(28),math.radians(332),ntheta); V=[]; F=[]; rings=[]
    for k,zz in enumerate(z):
        o=[]; inn=[]
        for t in th: o.append(len(V)); V.append([cx[k]+rx[k]*math.cos(t),cy[k]+ry[k]*math.sin(t),zz])
        for t in th: inn.append(len(V)); V.append([cx[k]+irx[k]*math.cos(t),cy[k]+iry[k]*math.sin(t),zz])
        rings.append((o,inn))
    for k in range(nz-1):
        o0,i0=rings[k]; o1,i1=rings[k+1]
        for j in range(ntheta-1):
            F += [[o0[j],o1[j],o1[j+1]],[o0[j],o1[j+1],o0[j+1]],[i0[j],i1[j+1],i1[j]],[i0[j],i0[j+1],i1[j+1]]]
        for j in (0,ntheta-1): F += [[o0[j],i0[j],i1[j]],[o0[j],i1[j],o1[j]]]
    return trimesh.Trimesh(vertices=np.array(V),faces=np.array(F),process=True)

scene=trimesh.Scene()
scene.add_geometry(paint(curved_cutaway_shaft(),CORTICAL),node_name="Femur_Cortical_Shaft_Cutaway")
parts=[
("Femoral_Neck",cyl_between([.05,0,3.75],[-.62,.03,4.75],.50,40),BONE),
("Femoral_Head",ellipsoid([.96,.91,.96],[-1.12,.03,5.20]),BONE),
("Greater_Trochanter",ellipsoid([.62,.53,1.05],[.60,.02,4.42]),BONE),
("Lesser_Trochanter",ellipsoid([.32,.31,.46],[-.02,-.38,4.02],3),BONE),
("Proximal_Metaphysis",ellipsoid([.78,.58,.95],[.05,0,4.05]),BONE),
("Distal_Metaphysis",ellipsoid([.92,.68,1.02],[0,0,-4.58]),BONE),
("Medial_Condyle",ellipsoid([.70,.76,.72],[-.48,.02,-5.18]),BONE),
("Lateral_Condyle",ellipsoid([.64,.72,.68],[.50,0,-5.12]),BONE),
("Patellar_Surface",ellipsoid([.46,.26,.48],[0,-.48,-5.03]),BONE),
("Head_Articular_Cartilage",ellipsoid([.99,.94,.99],[-1.14,.03,5.23]),CARTILAGE),
("Medial_Condyle_Cartilage",ellipsoid([.71,.77,.20],[-.48,.02,-5.68]),CARTILAGE),
("Lateral_Condyle_Cartilage",ellipsoid([.65,.73,.19],[.50,0,-5.60]),CARTILAGE),
("Yellow_Marrow",cyl_between([0,0,-3.9],[.03,0,3.55],.32,36),MARROW)]
for n,m,ma in parts: scene.add_geometry(paint(m,ma),node_name=n)

rng=np.random.default_rng(11)
def trab(center,scale,count,prefix):
    pts=[]
    for _ in range(count):
        while True:
            p=rng.uniform(-1,1,3)
            if np.dot(p,p)<1: pts.append(center+p*scale); break
    P=np.array(pts)
    for i in range(count*2):
        a=P[rng.integers(len(P))]; idx=np.argsort(np.linalg.norm(P-a,axis=1))[1:8]; b=P[int(rng.choice(idx))]
        scene.add_geometry(paint(cyl_between(a,b,.026,10),SPONGY),node_name=f"{prefix}_{i:03d}")
trab(np.array([-.48,0,4.85]),np.array([1.05,.62,1.05]),52,"Proximal_Trabecula")
trab(np.array([0,0,-4.85]),np.array([.92,.60,.72]),40,"Distal_Trabecula")

for n,p0,p1,r,ma in [
("Nutrient_Artery",[.35,-.72,-.65],[.10,-.10,.05],.045,ARTERY),
("Artery_Ascending",[.10,-.10,.05],[-.06,0,1.25],.028,ARTERY),
("Artery_Descending",[.10,-.10,.05],[.08,0,-1.45],.026,ARTERY),
("Nutrient_Vein",[.46,-.68,-.70],[.22,-.06,0],.034,VEIN)]:
    scene.add_geometry(paint(cyl_between(p0,p1,r,18),ma),node_name=n)

center=np.array([3.3,0,.4])
sample=trimesh.creation.cylinder(radius=1.0,height=2.2,sections=64); sample.apply_translation(center)
scene.add_geometry(paint(sample,BONE),node_name="Compact_Bone_Microstructure")
for ang in np.linspace(0,2*math.pi,9,endpoint=False):
    c=center+np.array([.58*math.cos(ang),.58*math.sin(ang),1.12])
    for rr in (.20,.14,.08):
        ring=trimesh.creation.annulus(r_min=rr-.012,r_max=rr+.012,height=.02,sections=32); ring.apply_translation(c)
        scene.add_geometry(paint(ring,CORTICAL),node_name="Concentric_Lamella")
    canal=trimesh.creation.cylinder(radius=.04,height=.30,sections=18); canal.apply_translation(c-np.array([0,0,.14]))
    scene.add_geometry(paint(canal,ARTERY),node_name="Haversian_Canal")
scene.export(OUT)
print(f"Created {OUT} with {len(scene.geometry)} geometries")
