"""Jello the capybara (shark onesie) - procedural Blender model + "morning routine" animation.

Usage (bpy python module or `blender -b -P`):
    python3 build_jello.py -- --save jello.blend
    python3 build_jello.py -- --still 100 --res 1280 720 --samples 64 --out renders/f100.png
    python3 build_jello.py -- --anim 1 312 --res 1280 720 --samples 48 --out frames/
"""
import bpy, bmesh, math, os, sys
import numpy as np
from mathutils import Vector, Matrix

# ----------------------------------------------------------------------------- config
TODO_TITLE = "Jen's To-Do"
TODO_ITEMS = ["Wake up Jello", "Answer emails", "Finish the big project",
              "Team meeting at 2", "Ice cream break"]
FPS = 24
END = 312
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_REG = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def arg(name, n=1, default=None):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1:i + 1 + n] if n > 1 else argv[i + 1]
    return default

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
COL = scene.collection
R = math.radians

# ----------------------------------------------------------------------------- helpers
def hx(h):
    h = h.lstrip('#')
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple((x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c) + (1.0,)

def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes.get('Principled BSDF')
    return m, nt, b

def add_bump(nt, b, scale=200.0, strength=0.3, detail=3.0, dist=0.02):
    tc = nt.nodes.new('ShaderNodeTexCoord')
    n = nt.nodes.new('ShaderNodeTexNoise')
    n.inputs['Scale'].default_value = scale
    n.inputs['Detail'].default_value = detail
    bp = nt.nodes.new('ShaderNodeBump')
    bp.inputs['Strength'].default_value = strength
    bp.inputs['Distance'].default_value = dist
    nt.links.new(tc.outputs['Object'], n.inputs['Vector'])
    nt.links.new(n.outputs[0], bp.inputs['Height'])
    nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])
    return tc

def fabric(name, col, scale=170.0, bump=0.45, sheen=1.0, rough=0.9):
    m, nt, b = new_mat(name)
    b.inputs['Base Color'].default_value = hx(col) if isinstance(col, str) else col
    b.inputs['Roughness'].default_value = rough
    b.inputs['Sheen Weight'].default_value = sheen
    b.inputs['Sheen Roughness'].default_value = 0.45
    b.inputs['Specular IOR Level'].default_value = 0.2
    add_bump(nt, b, scale, bump)
    return m

def plain(name, col, rough=0.5, metallic=0.0, coat=0.0, emit=0.0, spec=0.5):
    m, nt, b = new_mat(name)
    c = hx(col) if isinstance(col, str) else col
    b.inputs['Base Color'].default_value = c
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metallic
    b.inputs['Coat Weight'].default_value = coat
    b.inputs['Specular IOR Level'].default_value = spec
    if emit:
        b.inputs['Emission Color'].default_value = c
        b.inputs['Emission Strength'].default_value = emit
    return m

def math_node(nt, op, a, b=None):
    n = nt.nodes.new('ShaderNodeMath')
    n.operation = op
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]

def mix_color(nt, fac, a, b):
    n = nt.nodes.new('ShaderNodeMix')
    n.data_type = 'RGBA'
    nt.links.new(fac, n.inputs[0])
    n.inputs[6].default_value = a
    n.inputs[7].default_value = b
    return n.outputs[2]

def link_obj(o):
    COL.objects.link(o)
    return o

def empty(name, loc, parent=None):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.1
    link_obj(e)
    if parent is not None:
        e.parent = parent
        e.location = Vector(loc) - REST[parent.name]
    else:
        e.location = loc
    REST[name] = Vector(loc)
    return e

REST = {}

def attach(child, emp, local=False):
    """Parent `child` to empty. local=False keeps child's rest/world placement."""
    child.parent = emp
    if not local:
        child.matrix_parent_inverse = Matrix.Translation(-REST[emp.name])

def mesh_obj(name, V, F, mats, sub=2, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in V], [], F)
    me.update()
    if smooth:
        me.shade_smooth()
    o = bpy.data.objects.new(name, me)
    for m in (mats if isinstance(mats, (list, tuple)) else [mats]):
        me.materials.append(m)
    if sub:
        s = o.modifiers.new('sub', 'SUBSURF')
        s.levels = 1
        s.render_levels = sub
    return link_obj(o)

_QS = {}
def quadsphere(n):
    if n not in _QS:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=n, use_grid_fill=True)
        bm.verts.ensure_lookup_table()
        V = np.array([v.co[:] for v in bm.verts])
        V = np.tan(V * (math.pi / 4))          # more even distribution
        V /= np.linalg.norm(V, axis=1)[:, None]
        F = [[v.index for v in f.verts] for f in bm.faces]
        bm.free()
        _QS[n] = (V, F)
    return _QS[n]

# --- signed-distance modelling (smooth unions of ellipsoids, ray-projected onto quad spheres)
def ell(c, r, rot=None):
    c = np.array(c, float); r = np.array(r, float)
    Rm = None if rot is None else np.array(Matrix.Rotation(R(rot[0]), 3, 'X') @ Matrix.Rotation(R(rot[1]), 3, 'Y') @ Matrix.Rotation(R(rot[2]), 3, 'Z')) if False else None
    if rot is not None:
        from mathutils import Euler
        Rm = np.array(Euler([R(a) for a in rot]).to_matrix())
    def f(p):
        q = p - c
        if Rm is not None:
            q = q @ Rm
        return (np.linalg.norm(q / r, axis=-1) - 1.0) * r.min()
    return f

def union(fs, k=0.1):
    def f(p):
        d = fs[0](p)
        for g in fs[1:]:
            e = g(p)
            h = np.clip(0.5 + 0.5 * (e - d) / k, 0, 1)
            d = e * (1 - h) + d * h - k * h * (1 - h)
        return d
    return f

def project(f, o, D, tmax=3.0):
    o = np.array(o, float)
    lo = np.zeros(len(D)); hi = np.full(len(D), tmax)
    for _ in range(36):
        mid = (lo + hi) / 2
        ins = f(o + D * mid[:, None]) < 0
        lo = np.where(ins, mid, lo); hi = np.where(ins, hi, mid)
    return o + D * ((lo + hi) / 2)[:, None]

def surf(f, o, target):
    """Point + outward normal on surface f along ray from o toward target."""
    d = np.array(target, float) - np.array(o, float)
    d /= np.linalg.norm(d)
    p = project(f, o, d[None, :])[0]
    e = 1e-3
    g = np.array([f((p + np.eye(3)[i] * e)[None, :])[0] - f((p - np.eye(3)[i] * e)[None, :])[0] for i in range(3)])
    return p, g / np.linalg.norm(g)

def sdf_obj(name, f, o, mats, n=20, sub=2):
    V, F = quadsphere(n)
    return mesh_obj(name, project(f, o, V), F, mats, sub)

def ellip(name, c, r, mat, rot=(0, 0, 0), n=8, sub=2):
    V, F = quadsphere(n)
    o = mesh_obj(name, V * np.array(r, float), F, mat, sub)
    o.location = c
    o.rotation_euler = [R(a) for a in rot]
    return o

def torus(name, Rmaj, rmin, mat, ribs=0, nu=64, nv=14):
    V = []; F = []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        rr = rmin * (1 + (0.07 * math.sin(ribs * u) if ribs else 0))
        for j in range(nv):
            v = 2 * math.pi * j / nv
            V.append(((Rmaj + rr * math.cos(v)) * math.cos(u), (Rmaj + rr * math.cos(v)) * math.sin(u), rr * math.sin(v)))
            F.append([i * nv + j, ((i + 1) % nu) * nv + j, ((i + 1) % nu) * nv + (j + 1) % nv, i * nv + (j + 1) % nv])
    return mesh_obj(name, np.array(V), F, mat, sub=1)

def tube(name, pts, radius, mat, radii=None, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE')
    cu.dimensions = '3D'
    sp = cu.splines.new('NURBS')
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (p[0], p[1], p[2], 1.0)
        sp.points[i].radius = radii[i] if radii else 1.0
    sp.use_endpoint_u = not cyclic
    sp.use_cyclic_u = cyclic
    sp.order_u = min(4, len(pts))
    cu.bevel_depth = radius
    cu.bevel_resolution = 4
    cu.resolution_u = 12
    cu.use_fill_caps = True
    o = bpy.data.objects.new(name, cu)
    cu.materials.append(mat)
    return link_obj(o)

def box(name, size, loc, mat, bevel=0.02, rot=(0, 0, 0)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    o.location = loc
    o.rotation_euler = [R(a) for a in rot]
    if bevel:
        b = o.modifiers.new('bev', 'BEVEL'); b.width = bevel; b.segments = 3
    return link_obj(o)

def cyl(name, r, h, loc, mat, seg=48, r2=None, bevel=0.0, sub=0):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=r, radius2=r if r2 is None else r2, depth=h)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me); bm.free()
    me.materials.append(mat)
    me.shade_smooth()
    o = bpy.data.objects.new(name, me)
    o.location = loc
    if bevel:
        b = o.modifiers.new('bev', 'BEVEL'); b.width = bevel; b.segments = 3; b.limit_method = 'ANGLE'
    if sub:
        s = o.modifiers.new('sub', 'SUBSURF'); s.levels = 1; s.render_levels = sub
    return link_obj(o)

def text(name, body, size, mat, font, loc, extrude=0.004, align='LEFT'):
    cu = bpy.data.curves.new(name, 'FONT')
    cu.body = body
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.font = font
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    o.location = loc
    o.rotation_euler = (R(90), 0, 0)
    return link_obj(o)

def K(o, f, loc=None, rot=None, scl=None):
    if loc is not None:
        o.location = loc; o.keyframe_insert('location', frame=f)
    if rot is not None:
        o.rotation_euler = [R(a) for a in rot]; o.keyframe_insert('rotation_euler', frame=f)
    if scl is not None:
        o.scale = scl if hasattr(scl, '__len__') else (scl, scl, scl); o.keyframe_insert('scale', frame=f)

def pop(o, f, s=1.0, dur=8):
    """Poppy scale-in with overshoot."""
    K(o, f, scl=0.001); K(o, f + dur * 0.45, scl=s * 1.25); K(o, f + dur * 0.75, scl=s * 0.92); K(o, f + dur, scl=s)

# ----------------------------------------------------------------------------- materials
M_BLUE = fabric('fleece_blue', '#5384CF', bump=0.6)
M_CREAM = fabric('fleece_cream', '#F4ECDD', bump=0.35)
M_CUFF = fabric('cuff_blue', '#4B79C2', scale=120, bump=0.5)
M_TOOTH = plain('tooth', '#FBF8F0', rough=0.55)
M_EYE = plain('eye', '#050505', rough=0.04, coat=1.0)
M_WHITE = plain('highlight', '#FFFFFF', rough=0.3, emit=1.5)
M_DARK = plain('line_dark', '#3A2012', rough=0.7)
M_NOSTRIL = plain('nostril', '#4A2A16', rough=0.8)
M_GILL = fabric('gill', '#2C4270', bump=0.2)
M_MOUTH = plain('mouth_in', '#6B1F2A', rough=0.6)
M_TONGUE = plain('tongue', '#F08A9B', rough=0.45)
M_METAL = plain('zip_metal', '#C9CCD2', rough=0.25, metallic=1.0)

# skin: caramel velvet with darker muzzle
M_SKIN, nt, b = new_mat('skin')
b.inputs['Roughness'].default_value = 0.8
b.inputs['Sheen Weight'].default_value = 0.8
b.inputs['Sheen Roughness'].default_value = 0.4
b.inputs['Subsurface Weight'].default_value = 0.12
b.inputs['Subsurface Radius'].default_value = (0.9, 0.4, 0.2)
b.inputs['Subsurface Scale'].default_value = 0.08
b.inputs['Specular IOR Level'].default_value = 0.25
tc = add_bump(nt, b, 260, 0.3)
sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], sep.inputs[0])
mr = nt.nodes.new('ShaderNodeMapRange'); mr.interpolation_type = 'SMOOTHSTEP'
mr.inputs['From Min'].default_value = -0.40; mr.inputs['From Max'].default_value = -0.78
nt.links.new(sep.outputs['Y'], mr.inputs['Value'])
nt.links.new(mix_color(nt, mr.outputs[0], hx('#DD9C58'), hx('#AE6E3E')), b.inputs['Base Color'])
M_SKIN2 = fabric('skin_plain', '#CF8A48', scale=260, bump=0.3, sheen=0.8, rough=0.8)   # hands / feet / ears
M_EARIN = fabric('ear_inner', '#C4775C', scale=260, bump=0.2)

# body: blue fleece with cream belly panel
M_BODY, nt, b = new_mat('body_fleece')
b.inputs['Roughness'].default_value = 0.9
b.inputs['Sheen Weight'].default_value = 1.0
b.inputs['Sheen Roughness'].default_value = 0.45
b.inputs['Specular IOR Level'].default_value = 0.2
tc = add_bump(nt, b, 170, 0.6)
sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], sep.inputs[0])
ex = math_node(nt, 'POWER', math_node(nt, 'DIVIDE', sep.outputs['X'], 0.40), 2.0)
ez = math_node(nt, 'POWER', math_node(nt, 'DIVIDE', math_node(nt, 'SUBTRACT', sep.outputs['Z'], 0.50), 0.50), 2.0)
inside = math_node(nt, 'LESS_THAN', math_node(nt, 'ADD', ex, ez), 1.0)
front = math_node(nt, 'LESS_THAN', sep.outputs['Y'], -0.08)
nt.links.new(mix_color(nt, math_node(nt, 'MULTIPLY', inside, front), hx('#5384CF'), hx('#F4ECDD')), b.inputs['Base Color'])

# ----------------------------------------------------------------------------- JELLO
# Rest pose: standing, facing -Y, feet on z=0, about 2.0 units tall.
HIP_Z = 0.345
root = empty('jello_root', (0, 0, 0))
pelvis = empty('pelvis', (0, 0, HIP_Z), root)
torso = empty('torso', (0, 0, HIP_Z), pelvis)
head = empty('head', (0, 0.02, 1.04), torso)
armP = empty('arm_P', (0.43, -0.02, 0.98), torso)
armN = empty('arm_N', (-0.43, -0.02, 0.98), torso)
legP = empty('leg_P', (0.26, 0.0, HIP_Z), pelvis)
legN = empty('leg_N', (-0.26, 0.0, HIP_Z), pelvis)
tail = empty('tail', (0, 0.45, 0.45), torso)

# --- body
body_f = union([ell((0, 0.03, 0.60), (0.57, 0.53, 0.54)), ell((0, 0.0, 0.95), (0.43, 0.39, 0.34))], 0.22)
BODY_O = (0, 0, 0.68)
body = sdf_obj('body', body_f, BODY_O, M_BODY, n=28)
attach(body, torso)

# zipper down the belly
zp = []
for a in np.linspace(58, -62, 22):
    p, n = surf(body_f, BODY_O, (0, BODY_O[1] - math.cos(R(a)) * 2, BODY_O[2] + math.sin(R(a)) * 2))
    zp.append(p + n * 0.004)
zipper = tube('zipper', zp, 0.016, M_CREAM)
attach(zipper, torso)
pull = box('zip_pull', (0.05, 0.016, 0.085), Vector(zp[3]) + Vector((0, -0.022, -0.02)), M_METAL, bevel=0.006, rot=(-12, 0, 0))
attach(pull, torso)
slider = box('zip_slider', (0.06, 0.03, 0.05), Vector(zp[2]) + Vector((0, -0.012, 0)), M_METAL, bevel=0.008, rot=(-25, 0, 0))
attach(slider, torso)

# dorsal fin
V, F = quadsphere(10)
t = (V[:, 2] + 1) / 2
fin_v = np.stack([0.075 * V[:, 0] * (1 - 0.55 * t), 0.24 * V[:, 1] * (1 - 0.7 * t) + 0.20 * t ** 2, 0.30 * V[:, 2]], 1)
fin = mesh_obj('dorsal_fin', fin_v, F, M_BLUE)
fin.location = (0, 0.60, 1.0); fin.rotation_euler = (R(-62), 0, 0)
attach(fin, torso)

# tail (stalk + crescent flukes)
tail_f = union([ell((0, 0.50, 0.43), (0.24, 0.30, 0.24)), ell((0, 0.86, 0.50), (0.12, 0.26, 0.13))], 0.15)
stalk = sdf_obj('tail_stalk', tail_f, (0, 0.62, 0.46), M_BLUE, n=12)
fl1 = ellip('tail_fluke_up', (0, 1.16, 0.80), (0.055, 0.15, 0.36), M_BLUE, rot=(-32, 0, 0))
fl2 = ellip('tail_fluke_dn', (0, 1.12, 0.30), (0.055, 0.13, 0.27), M_BLUE, rot=(38, 0, 0))
for o in (stalk, fl1, fl2):
    attach(o, tail)

# --- arms (built along local -Z from the shoulder)
def build_arm(emp, sx):
    f = union([ell((0, 0, -0.02), (0.20, 0.21, 0.20)), ell((0, 0, -0.24), (0.175, 0.185, 0.30))], 0.08)
    sl = sdf_obj(emp.name + '_sleeve', f, (0, 0, -0.2), M_BLUE, n=12)
    cu = torus(emp.name + '_cuff', 0.118, 0.055, M_CUFF, ribs=26)
    cu.location = (0, 0, -0.50)
    hf = union([ell((0, 0, -0.61), (0.125, 0.135, 0.12))] +
               [ell((dx, -0.045, -0.70), (0.05, 0.062, 0.058)) for dx in (-0.075, 0.0, 0.075)] +
               [ell((-sx * 0.11, -0.03, -0.60), (0.05, 0.055, 0.06))], 0.035)
    hd = sdf_obj(emp.name + '_hand', hf, (0, 0, -0.61), M_SKIN2, n=12)
    for o in (sl, cu, hd):
        attach(o, emp, local=True)
build_arm(armP, 1); build_arm(armN, -1)

# --- legs (built along local -Z from the hip)
def build_leg(emp):
    pl = ellip(emp.name + '_pant', (0, 0, -0.07), (0.22, 0.23, 0.2), M_BLUE)
    cu = torus(emp.name + '_cuff', 0.15, 0.055, M_CUFF, ribs=30)
    cu.location = (0, 0, -0.20)
    ff = union([ell((0, -0.07, -0.262), (0.175, 0.24, 0.083))] +
               [ell((dx, -0.285, -0.268), (0.066, 0.075, 0.066)) for dx in (-0.105, 0.0, 0.105)], 0.03)
    ft = sdf_obj(emp.name + '_foot', ff, (0, -0.07, -0.262), M_SKIN2, n=14)
    for o in (pl, cu, ft):
        attach(o, emp, local=True)
build_leg(legP); build_leg(legN)

# --- head
HEAD_O = (0, 0.0, 1.34)
head_f = union([ell((0, 0.03, 1.36), (0.50, 0.47, 0.44)),
                ell((0, -0.40, 1.24), (0.39, 0.40, 0.335)),
                ell((0, -0.12, 1.14), (0.43, 0.40, 0.28))], 0.12)
head_m = sdf_obj('head_skin', head_f, HEAD_O, M_SKIN, n=28)
attach(head_m, head)

def on_head(target, lift=0.0):
    p, n = surf(head_f, HEAD_O, target)
    return Vector(p + n * lift), Vector(n)

# eyes: open bead + closed arc, both scalable about the eye centre
EYES = {}
for sx, tag in ((1, 'P'), (-1, 'N')):
    p, n = on_head((sx * 0.47, -0.60, 1.49))
    c = p - n * 0.03
    eo = empty('eye_open_' + tag, c, head)
    ec = empty('eye_closed_' + tag, c, head)
    ball = ellip('eyeball_' + tag, c, (0.09, 0.09, 0.098), M_EYE, n=8)
    attach(ball, eo)
    up = (Vector((0, 0, 1)) - n * n.z).normalized()
    side = n.cross(up) * (1 if sx > 0 else -1)            # points toward the back of the head (outer corner)
    hl = ellip('eye_glint_' + tag, c + n * 0.066 + up * 0.03 - side * 0.03, (0.022,) * 3, M_WHITE, n=4)
    attach(hl, eo)
    # lashes
    for i, a in enumerate((8, 34, 60)):
        base = c + n * 0.045 + (side * math.cos(R(a)) + up * math.sin(R(a))) * 0.078
        mid = base + (side * math.cos(R(a + 8)) + up * math.sin(R(a + 8))) * 0.045 + n * 0.012
        tip = base + (side * math.cos(R(a + 22)) + up * math.sin(R(a + 22))) * 0.085 + n * 0.012
        attach(tube(f'lash_{tag}{i}', [base, mid, tip], 0.012, M_DARK, radii=[1.0, 0.8, 0.25]), head)
    # brow
    bpts = []
    for a in np.linspace(55, 115, 6):
        q, qn = on_head(c + (side * math.cos(R(a)) + up * math.sin(R(a))) * 0.19 + n * 0.3, 0.004)
        bpts.append(q)
    attach(tube('brow_' + tag, bpts, 0.011, M_DARK, radii=[0.5, 1, 1, 1, 0.8, 0.4]), head)
    # closed-eye arc (u-shape; flip Z for a happy arc)
    cpts = [c + n * 0.05 + (side * math.cos(R(a)) - up * math.sin(R(a)) * 0.75) * 0.085 for a in np.linspace(15, 165, 8)]
    arc = tube('eye_arc_' + tag, cpts, 0.013, M_DARK)
    attach(arc, ec)
    EYES[tag] = (eo, ec)

# nostrils
for sx in (1, -1):
    p, n = on_head((sx * 0.20, -1.2, 1.44))
    o = ellip('nostril', p - n * 0.012, (0.045, 0.03, 0.034), M_NOSTRIL, n=5, rot=(0, sx * 18, 0))
    attach(o, head)

# philtrum + smile
ph = [on_head((0, -1.2, z), 0.002)[0] for z in np.linspace(1.25, 1.07, 6)]
attach(tube('philtrum', ph, 0.010, M_DARK), head)
sm = [on_head((x, -1.2, 1.07 + 0.045 * (abs(x) / 0.26) ** 2), 0.002)[0] for x in np.linspace(-0.26, 0.26, 11)]
attach(tube('smile', sm, 0.011, M_DARK, radii=[0.4] + [1] * 9 + [0.4]), head)

# open mouth (yawn / cheer) - scaled up from ~0
mp, mn = on_head((0, -1.2, 1.00))
mouthO = empty('mouth_open', mp - mn * 0.012, head)
m_up = (Vector((0, 0, 1)) - mn * mn.z).normalized()
attach(ellip('mouth_cavity', mp - mn * 0.012, (0.17, 0.05, 0.14), M_MOUTH, n=6, rot=(26, 0, 0)), mouthO)
attach(ellip('tongue', mp + mn * 0.012 - m_up * 0.055, (0.09, 0.035, 0.055), M_TONGUE, n=6, rot=(26, 0, 0)), mouthO)

# --- shark hood: polar grid around the face-opening axis, projected on the hood surface
HOOD_O = np.array((0, 0.03, 1.36))
hood_f = union([ell((0, 0.06, 1.38), (0.655, 0.63, 0.60)),
                ell((0, -0.30, 1.79), (0.40, 0.44, 0.21), rot=(-8, 0, 0)),
                ell((0, 0.12, 0.98), (0.50, 0.46, 0.30))], 0.16)
AX = np.array((0, -1.0, -0.30)); AX /= np.linalg.norm(AX)
UX = np.array((1.0, 0, 0)); VX = np.cross(UX, AX); VX /= np.linalg.norm(VX)
if VX[2] < 0:
    VX = -VX
NR, NS = 26, 72
def rim_angle(phi):
    s = math.sin(phi)
    return R(76 + (14 * (-s) ** 1.5 if s < 0 else -13 * s))
dirs = []
for i in range(NR + 1):
    for j in range(NS):
        phi = 2 * math.pi * j / NS
        t0 = rim_angle(phi)
        th = t0 + (math.pi - 0.02 - t0) * (i / NR)
        dirs.append(math.cos(th) * AX + math.sin(th) * (math.cos(phi) * UX + math.sin(phi) * VX))
dirs = np.array(dirs)
HV = project(hood_f, HOOD_O, dirs)
HF = [[i * NS + j, (i + 1) * NS + j, (i + 1) * NS + (j + 1) % NS, i * NS + (j + 1) % NS] for i in range(NR) for j in range(NS)]
HF.append([NR * NS + j for j in range(NS)])
hood = mesh_obj('hood', HV, HF, [M_BLUE, M_CREAM], sub=0)
for pi_ in range(2 * NS):
    hood.data.polygons[pi_].material_index = 1          # cream 'gum' band around the opening
so = hood.modifiers.new('solid', 'SOLIDIFY')
so.thickness = 0.085; so.offset = -1.0; so.material_offset = 1; so.material_offset_rim = 1
ss = hood.modifiers.new('sub', 'SUBSURF'); ss.levels = 1; ss.render_levels = 2
attach(hood, head)

# teeth along the upper rim
rim = HV[:NS]
for j in range(NS):
    phi = 2 * math.pi * j / NS
    if not (R(8) < phi < R(172)) or j % 3 != 0:
        continue
    p = Vector(rim[j])
    rad = p - Vector(HOOD_O); rad = rad - Vector(AX) * rad.dot(Vector(AX))
    d = (-rad.normalized() * 1.0 + Vector(AX) * 0.25).normalized()
    hgt = 0.17
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=12, radius1=0.07, radius2=0.012, depth=hgt)
    me = bpy.data.meshes.new('tooth'); bm.to_mesh(me); bm.free(); me.shade_smooth(); me.materials.append(M_TOOTH)
    to = bpy.data.objects.new('tooth', me); link_obj(to)
    s = to.modifiers.new('sub', 'SUBSURF'); s.levels = 1; s.render_levels = 2
    to.rotation_mode = 'QUATERNION'
    to.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    to.location = p - Vector(AX) * 0.05 + d * (hgt * 0.42)
    attach(to, head)

def on_hood(target, lift=0.0):
    p, n = surf(hood_f, HOOD_O, target)
    return Vector(p + n * lift), Vector(n)

# shark eyes + gills + capybara ears poking out of the hood
for sx in (1, -1):
    p, n = on_hood((sx * 0.55, -0.62, 2.05))
    attach(ellip('shark_eye', p - n * 0.02, (0.062,) * 3, M_EYE, n=6), head)
    attach(ellip('shark_glint', p + n * 0.03 + Vector((0, -0.012, 0.022)), (0.018,) * 3, M_WHITE, n=4), head)
    for i in range(3):
        p, n = on_hood((sx * 1.0, -0.28 + 0.13 * i, 1.50 - 0.02 * i))
        g = ellip('gill', p - n * 0.008, (0.022, 0.026, 0.085), M_GILL, n=5, rot=(-22, 0, 0))
        attach(g, head)
    p, n = on_hood((sx * 0.85, 0.02, 2.02))
    attach(ellip('ear', p + n * 0.045, (0.155, 0.085, 0.155), M_SKIN2, n=8), head)
    attach(ellip('ear_inner', p + n * 0.05 + Vector((0, -0.055, 0)), (0.095, 0.045, 0.095), M_EARIN, n=6), head)

# ----------------------------------------------------------------------------- SET
BED = Vector((-3.35, 0.7, 0))
SEAT = Vector((1.55, 0.25, 0))
SEAT_YAW = -68.0

# floor (oak planks) / wall / baseboard / rug
M_FLOOR, nt, b = new_mat('floor_oak')
tc = nt.nodes.new('ShaderNodeTexCoord')
mp_ = nt.nodes.new('ShaderNodeMapping'); mp_.inputs['Scale'].default_value = (1.6, 9.0, 1); mp_.inputs['Rotation'].default_value = (0, 0, R(90))
nt.links.new(tc.outputs['Object'], mp_.inputs[0])
br = nt.nodes.new('ShaderNodeTexBrick')
br.inputs['Color1'].default_value = hx('#C99A66'); br.inputs['Color2'].default_value = hx('#B98753'); br.inputs['Mortar'].default_value = hx('#7A5632')
br.inputs['Scale'].default_value = 1.0; br.inputs['Mortar Size'].default_value = 0.006; br.inputs['Brick Width'].default_value = 2.2; br.inputs['Row Height'].default_value = 1.0
br.offset = 0.37
nt.links.new(mp_.outputs[0], br.inputs['Vector'])
gr = nt.nodes.new('ShaderNodeTexNoise'); gr.inputs['Scale'].default_value = 3.0; gr.inputs['Detail'].default_value = 8
mp2 = nt.nodes.new('ShaderNodeMapping'); mp2.inputs['Scale'].default_value = (6, 160, 1)
nt.links.new(tc.outputs['Object'], mp2.inputs[0]); nt.links.new(mp2.outputs[0], gr.inputs['Vector'])
mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'; mx.inputs[0].default_value = 0.35
nt.links.new(br.outputs['Color'], mx.inputs[6]); nt.links.new(gr.outputs[1], mx.inputs[7])
nt.links.new(mx.outputs[2], b.inputs['Base Color'])
b.inputs['Roughness'].default_value = 0.38
bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = 0.25
nt.links.new(br.outputs['Fac'], bp.inputs['Height']); bp.invert = True
nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])

def plane(name, sx, sy, loc, mat, rot=(0, 0, 0)):
    V = [(-sx / 2, -sy / 2, 0), (sx / 2, -sy / 2, 0), (sx / 2, sy / 2, 0), (-sx / 2, sy / 2, 0)]
    o = mesh_obj(name, np.array(V), [[0, 1, 2, 3]], mat, sub=0, smooth=False)
    o.location = loc; o.rotation_euler = [R(a) for a in rot]
    return o

plane('floor', 60, 60, (0, 0, 0), M_FLOOR)
M_WALL, nt, b = new_mat('wall'); b.inputs['Base Color'].default_value = hx('#F6DDCB'); b.inputs['Roughness'].default_value = 0.9
add_bump(nt, b, 60, 0.08)
plane('wall', 60, 30, (0, 3.6, 10), M_WALL, rot=(90, 0, 0))
box('baseboard', (60, 0.08, 0.32), (0, 3.56, 0.16), plain('baseboard', '#FFFBF4', rough=0.4), bevel=0.02)
M_RUG = fabric('rug', '#F7E3D4', scale=60, bump=1.0, rough=1.0)
cyl('rug', 2.6, 0.05, (0.6, 0.7, 0.025), M_RUG, seg=96, bevel=0.02)
cyl('rug_ring', 2.25, 0.052, (0.6, 0.7, 0.026), fabric('rug2', '#F2B8B0', scale=60, bump=1.0, rough=1.0), seg=96)
cyl('rug_in', 2.05, 0.054, (0.6, 0.7, 0.027), M_RUG, seg=96)

# round pet bed
M_BEDR = fabric('bed_ring', '#F29CAE', scale=90, bump=0.7)
bed_ring = torus('bed_ring', 1.25, 0.34, M_BEDR, nu=72, nv=20)
bed_ring.location = BED + Vector((0, 0, 0.34)); bed_ring.scale = (1, 1, 0.95)
bed_cush = ellip('bed_cushion', BED + Vector((0, 0, 0.2)), (1.2, 1.2, 0.26), fabric('bed_cushion', '#FFF4E6', scale=90, bump=0.6), n=12)

# floor cushion to sit on
seat_c = ellip('seat_cushion', SEAT + Vector((0, 0, 0.11)), (0.82, 0.82, 0.13), fabric('seat', '#F5B93F', scale=110, bump=0.6), n=12)

# desk group in Jello's sitting frame (she sits at origin facing -Y)
desk = empty('desk', SEAT)
desk.rotation_euler = (0, 0, R(SEAT_YAW))
M_WOOD, nt, b = new_mat('desk_wood'); b.inputs['Base Color'].default_value = hx('#E7C79B'); b.inputs['Roughness'].default_value = 0.35
tcw = nt.nodes.new('ShaderNodeTexCoord'); mpw = nt.nodes.new('ShaderNodeMapping'); mpw.inputs['Scale'].default_value = (2, 30, 2)
nw = nt.nodes.new('ShaderNodeTexNoise'); nw.inputs['Scale'].default_value = 2.5; nw.inputs['Detail'].default_value = 6
nt.links.new(tcw.outputs['Object'], mpw.inputs[0]); nt.links.new(mpw.outputs[0], nw.inputs['Vector'])
nt.links.new(mix_color(nt, nw.outputs[0], hx('#F0D2A6'), hx('#D2A873')), b.inputs['Base Color'])
TOP = 0.52
DY = -1.02                                       # desk centre in front of her
parts = [box('desk_top', (1.5, 0.86, 0.06), (0, DY, TOP - 0.03), M_WOOD, bevel=0.025)]
for dx in (-0.64, 0.64):
    for dy in (-0.33, 0.33):
        parts.append(box('desk_leg', (0.08, 0.08, TOP - 0.06), (dx, DY + dy, (TOP - 0.06) / 2), M_WOOD, bevel=0.015))
# laptop
M_ALU = plain('aluminium', '#C5C8CF', rough=0.32, metallic=1.0)
LY = -0.93
parts.append(box('laptop_base', (0.74, 0.50, 0.03), (0, LY, TOP + 0.015), M_ALU, bevel=0.012))
M_KEYS, nt, b = new_mat('keys'); b.inputs['Roughness'].default_value = 0.5
tck = nt.nodes.new('ShaderNodeTexCoord'); brk = nt.nodes.new('ShaderNodeTexBrick')
brk.inputs['Color1'].default_value = hx('#2B2D33'); brk.inputs['Color2'].default_value = hx('#33353C'); brk.inputs['Mortar'].default_value = hx('#9A9DA5')
brk.inputs['Scale'].default_value = 22; brk.inputs['Mortar Size'].default_value = 0.012; brk.inputs['Brick Width'].default_value = 0.5; brk.inputs['Row Height'].default_value = 0.5
nt.links.new(tck.outputs['Object'], brk.inputs['Vector']); nt.links.new(brk.outputs['Color'], b.inputs['Base Color'])
parts.append(box('laptop_keys', (0.64, 0.24, 0.006), (0, LY - 0.08, TOP + 0.031), M_KEYS, bevel=0))
parts.append(box('laptop_pad', (0.22, 0.13, 0.004), (0, LY + 0.15, TOP + 0.031), plain('pad', '#B4B7BE', rough=0.25, metallic=1.0), bevel=0))
for o in parts:
    attach(o, desk, local=True)
lid = empty('laptop_lid', (0, 0, 0))
lid.parent = desk; lid.location = (0, LY - 0.25, TOP + 0.034)
lp = box('lid_shell', (0.74, 0.50, 0.022), (0, 0.25, 0.0), M_ALU, bevel=0.01)
M_SCREEN, nt, b = new_mat('screen'); b.inputs['Base Color'].default_value = hx('#101418'); b.inputs['Roughness'].default_value = 0.15
tcs = nt.nodes.new('ShaderNodeTexCoord'); sps = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tcs.outputs['Object'], sps.inputs[0])
mrs = nt.nodes.new('ShaderNodeMapRange'); mrs.inputs['From Min'].default_value = -0.25; mrs.inputs['From Max'].default_value = 0.25
nt.links.new(sps.outputs['Y'], mrs.inputs['Value'])
nt.links.new(mix_color(nt, mrs.outputs[0], hx('#BFE3FF'), hx('#FFD9E6')), b.inputs['Emission Color'])
SCREEN_E = b.inputs['Emission Strength']; SCREEN_E.default_value = 0.0
sc = box('lid_screen', (0.68, 0.44, 0.004), (0, 0.25, -0.012), M_SCREEN, bevel=0)
stk = cyl('lid_sticker', 0.085, 0.004, (0.0, 0.25, 0.012), plain('sticker', '#F58CA6', rough=0.4), seg=40)
for o in (lp, sc, stk):
    attach(o, lid, local=True)

# strawberry ice cream waiting in a cup on the desk
M_CONE, nt, b = new_mat('waffle'); b.inputs['Base Color'].default_value = hx('#E0A04A'); b.inputs['Roughness'].default_value = 0.6
tcc = nt.nodes.new('ShaderNodeTexCoord'); wv = nt.nodes.new('ShaderNodeTexChecker'); wv.inputs['Scale'].default_value = 34
bpc = nt.nodes.new('ShaderNodeBump'); bpc.inputs['Strength'].default_value = 0.6
nt.links.new(tcc.outputs['Object'], wv.inputs['Vector']); nt.links.new(wv.outputs['Fac'], bpc.inputs['Height']); nt.links.new(bpc.outputs['Normal'], b.inputs['Normal'])
M_ICE, nt, b = new_mat('strawberry'); b.inputs['Roughness'].default_value = 0.45; b.inputs['Subsurface Weight'].default_value = 0.3
b.inputs['Subsurface Radius'].default_value = (1, 0.4, 0.4); b.inputs['Subsurface Scale'].default_value = 0.05
tci = nt.nodes.new('ShaderNodeTexCoord'); ni = nt.nodes.new('ShaderNodeTexNoise'); ni.inputs['Scale'].default_value = 14; ni.inputs['Detail'].default_value = 5
vi = nt.nodes.new('ShaderNodeTexVoronoi'); vi.inputs['Scale'].default_value = 16
nt.links.new(tci.outputs['Object'], ni.inputs['Vector']); nt.links.new(tci.outputs['Object'], vi.inputs['Vector'])
spk = math_node(nt, 'LESS_THAN', vi.outputs['Distance'], 0.16)
nt.links.new(mix_color(nt, spk, hx('#F9A3B4'), hx('#E2586F')), b.inputs['Base Color'])
bpi = nt.nodes.new('ShaderNodeBump'); bpi.inputs['Strength'].default_value = 0.9; bpi.inputs['Distance'].default_value = 0.05
nt.links.new(ni.outputs[0], bpi.inputs['Height']); nt.links.new(bpi.outputs['Normal'], b.inputs['Normal'])
IC = Vector((-0.47, DY - 0.30, TOP))
ice = [cyl('cup', 0.085, 0.14, IC + Vector((0, 0, 0.07)), plain('cup', '#FFFFFF', rough=0.2, coat=0.5), r2=0.105, seg=40, bevel=0.01),
       cyl('cone', 0.012, 0.36, IC + Vector((0, 0, 0.22)), M_CONE, r2=0.105, seg=40),
       ellip('scoop', IC + Vector((0, 0, 0.45)), (0.135, 0.135, 0.12), M_ICE, n=10),
       torus('scoop_skirt', 0.105, 0.035, M_ICE, nu=40, nv=10)]
ice[3].location = IC + Vector((0, 0, 0.40))
for o in ice:
    attach(o, desk, local=True)

# --- to-do board (pops up next to the desk)
board = empty('todo_board', (-0.85, 1.5, 0))
board.rotation_euler = (R(-7), 0, R(14))
FB = bpy.data.fonts.load(FONT_BOLD); FR = bpy.data.fonts.load(FONT_REG)
M_INK = plain('ink', '#2B2A33', rough=0.6)
M_PAPER, nt, b = new_mat('paper'); b.inputs['Base Color'].default_value = hx('#FFFDF6'); b.inputs['Roughness'].default_value = 0.75
add_bump(nt, b, 400, 0.05)
M_PINK = plain('pink', '#F0607F', rough=0.45)
M_GREEN = plain('check_green', '#2FB86A', rough=0.4, emit=0.25)
BW, BH, BZ = 1.78, 1.86, 0.52
bp_ = [box('board_back', (BW, 0.05, BH), (0, 0, BZ + BH / 2), M_WOOD, bevel=0.04),
       box('board_paper', (BW - 0.16, 0.012, BH - 0.2), (0, -0.03, BZ + BH / 2 - 0.03), M_PAPER, bevel=0.004),
       box('board_clip', (0.46, 0.05, 0.13), (0, -0.045, BZ + BH - 0.07), M_METAL, bevel=0.02),
       box('board_legL', (0.07, 0.07, BZ + 0.3), (-0.62, 0.03, (BZ + 0.3) / 2), M_WOOD, bevel=0.015),
       box('board_legR', (0.07, 0.07, BZ + 0.3), (0.62, 0.03, (BZ + 0.3) / 2), M_WOOD, bevel=0.015),
       box('board_strut', (0.07, 0.07, 1.7), (0, 0.38, 0.82), M_WOOD, bevel=0.015, rot=(24, 0, 0))]
TY = -0.04
bp_.append(text('todo_title', TODO_TITLE, 0.215, M_PINK, FB, (0, TY, BZ + BH - 0.42), align='CENTER'))
bp_.append(box('todo_rule', (1.2, 0.006, 0.014), (0, TY, BZ + BH - 0.50), M_PINK, bevel=0))
CHECKS = []
for i, item in enumerate(TODO_ITEMS):
    z = BZ + BH - 0.76 - i * 0.235
    bp_.append(text(f'todo_{i}', item, 0.124, M_INK, FR, (-0.50, TY, z)))
    for (sz, off) in (((0.12, 0.006, 0.014), (0, 0, 0.053)), ((0.12, 0.006, 0.014), (0, 0, -0.053)),
                      ((0.014, 0.006, 0.12), (0.053, 0, 0)), ((0.014, 0.006, 0.12), (-0.053, 0, 0))):
        bp_.append(box('todo_box', sz, Vector((-0.66, TY, z + 0.04)) + Vector(off), M_INK, bevel=0))
    ce = empty(f'check_{i}', (0, 0, 0)); ce.parent = board; ce.location = (-0.655, TY - 0.012, z + 0.04)
    ck = tube(f'checkmark_{i}', [(-0.055, 0, 0.005), (-0.015, 0, -0.045), (0.085, 0, 0.085)], 0.017, M_GREEN)
    ck.data.splines[0].order_u = 2
    attach(ck, ce, local=True)
    CHECKS.append(ce)
for o in bp_:
    attach(o, board, local=True)

# --- poppy bits: sleepy z's and sparkles
M_Z = plain('zzz', '#BFD6FF', rough=0.4, emit=1.6)
M_STAR = plain('sparkle', '#FFD54A', rough=0.3, emit=2.0)
def star(name, r=0.12):
    V = [(0, 0, 0.03), (0, 0, -0.03)]; F = []
    for i in range(8):
        a = math.pi / 4 * i
        rr = r if i % 2 == 0 else r * 0.3
        V.append((rr * math.cos(a), rr * math.sin(a), 0))
    for i in range(8):
        F.append([0, 2 + i, 2 + (i + 1) % 8]); F.append([1, 2 + (i + 1) % 8, 2 + i])
    o = mesh_obj(name, np.array(V), F, M_STAR, sub=0, smooth=False)
    o.rotation_euler = (R(90), 0, 0)
    o.scale = (0.001,) * 3
    return o

# ----------------------------------------------------------------------------- lights / world / camera
world = bpy.data.worlds.new('world'); scene.world = world; world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = hx('#DCE9FF')
WORLD_E = bg.inputs['Strength']

sun_d = bpy.data.lights.new('sun', 'SUN'); sun_d.color = hx('#FFE2B8')[:3]; sun_d.angle = R(2.5)
sun = link_obj(bpy.data.objects.new('sun', sun_d))
sun.rotation_euler = (R(52), R(-8), R(-38))
sdir = (sun.rotation_euler.to_matrix() @ Vector((0, 0, -1))).normalized()
# window gobo: a big shadow card with window panes cut out, between the sun and the room
gs = 0.12; gn = 150
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=gn, y_segments=gn, size=gs * gn / 2)
kill = []
for fc in bm.faces:
    c = fc.calc_center_median()
    if abs(c.x) < 4.6 and abs(c.y) < 2.5:
        u = (c.x + 4.6) % 2.3; v = (c.y + 2.5) % 2.5
        if 0.14 < u < 2.16 and 0.14 < v < 2.36:
            kill.append(fc)
bmesh.ops.delete(bm, geom=kill, context='FACES')
gm = bpy.data.meshes.new('gobo'); bm.to_mesh(gm); bm.free()
gobo = link_obj(bpy.data.objects.new('window_gobo', gm))
gm.materials.append(plain('gobo', '#000000'))
gobo.rotation_euler = sun.rotation_euler
gobo.location = Vector((-0.6, 1.2, 1.2)) - sdir * 14
gobo.visible_camera = False; gobo.visible_diffuse = False; gobo.visible_glossy = False

def area(name, loc, target, size, energy, col):
    d = bpy.data.lights.new(name, 'AREA'); d.size = size; d.energy = energy; d.color = hx(col)[:3]
    o = link_obj(bpy.data.objects.new(name, d)); o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return o
key = area('key', (-3.5, -6.5, 5.5), (0, 0.5, 0.8), 7.0, 2600, '#FFF1E0')
rim_l = area('rim', (5.5, 2.6, 4.0), (0, 0.3, 1.0), 4.0, 900, '#CFE0FF')

cam_d = bpy.data.cameras.new('cam'); cam_d.lens = 42; cam_d.sensor_width = 36
cam = link_obj(bpy.data.objects.new('cam', cam_d)); scene.camera = cam
aim = empty('cam_aim', (0, 0, 1))
tr = cam.constraints.new('TRACK_TO'); tr.target = aim; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
cam_d.dof.use_dof = True; cam_d.dof.focus_object = head; cam_d.dof.aperture_fstop = 4.0

# ----------------------------------------------------------------------------- ANIMATION
ALL_J = [root, pelvis, torso, head, armP, armN, legP, legN, tail]
BEDP = BED + Vector((0.25, -0.21, 0))            # where her hips rest in the bed
BED_YAW = 50.0
def eyes(f, state):
    """state: 'open' | 'half' | 'sleep' (u arcs) | 'happy' (n arcs)"""
    for eo, ec in EYES.values():
        if state == 'open':
            K(eo, f, scl=1.0); K(ec, f, scl=0.001)
        elif state == 'half':
            K(eo, f, scl=(1.0, 1.0, 0.45)); K(ec, f, scl=0.001)
        elif state == 'sleep':
            K(eo, f, scl=0.001); K(ec, f, scl=(1, 1, 1))
        else:
            K(eo, f, scl=0.001); K(ec, f, scl=(1, 1, -1))
def blink(f):
    eyes(f - 1, 'open'); eyes(f + 1, 'sleep'); eyes(f + 3, 'sleep'); eyes(f + 5, 'open')

ARM_REST_P, ARM_REST_N = (-12, -14, 0), (-12, 14, 0)

# ---- 1. asleep in the bed (reclined against the ring), breathing, z's popping
LIE = -62
K(root, 1, loc=BEDP, rot=(0, 0, BED_YAW), scl=1.0)
K(pelvis, 1, loc=(0, 0, 0.72), rot=(0, 0, 0))
K(torso, 1, rot=(LIE, 0, 0), scl=1.0)
K(head, 1, rot=(14, 0, 8))
K(legP, 1, rot=(-78, 0, 24)); K(legN, 1, rot=(-78, 0, -24))
K(armP, 1, rot=(-48, 28, 0)); K(armN, 1, rot=(-48, -28, 0))
K(tail, 1, rot=(0, 0, 0))
K(mouthO, 1, scl=0.001)
eyes(1, 'sleep')
for f in range(1, 60, 18):                        # breathing
    K(torso, f, scl=(1, 1, 1)); K(torso, f + 9, scl=(1.035, 1.05, 1.0))
    K(armP, f, rot=(-48, 28, 0)); K(armP, f + 9, rot=(-52, 28, 0))
    K(armN, f, rot=(-48, -28, 0)); K(armN, f + 9, rot=(-52, -28, 0))
zs = []
for i, f0 in enumerate((4, 20, 36)):
    z = text(f'zzz_{i}', 'z' if i < 2 else 'Z', 0.34 + 0.09 * i, M_Z, FB, (0, 0, 0), extrude=0.02, align='CENTER')
    base = BED + Vector((-0.45, -0.9, 1.72))
    K(z, 1, loc=base, scl=0.001); K(z, f0, loc=base, scl=0.001, rot=(90, 0, -12))
    K(z, f0 + 5, loc=base + Vector((0.1, 0, 0.18)), scl=1.2, rot=(90, 0, 10))
    K(z, f0 + 16, loc=base + Vector((0.3, 0, 0.62)), scl=1.0, rot=(90, 0, -10))
    K(z, f0 + 24, loc=base + Vector((0.42, 0, 0.9)), scl=0.001, rot=(90, 0, 12))

# ---- 2. wake up: sleepy flutter, pop awake
K(head, 54, rot=(14, 0, 8))
eyes(55, 'sleep'); eyes(58, 'half'); eyes(62, 'half'); eyes(64, 'sleep'); eyes(68, 'sleep'); eyes(71, 'half'); eyes(76, 'open')
K(head, 64, rot=(6, 0, -4)); K(head, 76, rot=(-4, 0, 0))
K(torso, 60, rot=(LIE, 0, 0), scl=1.0)
K(pelvis, 60, loc=(0, 0, 0.72))
# sit up
K(torso, 72, rot=(LIE + 6, 0, 0)); K(torso, 84, rot=(10, 0, 0), scl=(1.03, 1.03, 0.96)); K(torso, 90, rot=(0, 0, 0), scl=1.0)
K(pelvis, 72, loc=(0, 0, 0.72)); K(pelvis, 86, loc=(0, 0, 0.70))
K(legP, 72, rot=(-78, 0, 24)); K(legP, 86, rot=(-86, 0, 30))
K(legN, 72, rot=(-78, 0, -24)); K(legN, 86, rot=(-86, 0, -30))
K(armP, 64, rot=(-48, 28, 0)); K(armP, 86, rot=(-30, -10, 0)); K(armN, 64, rot=(-48, -28, 0)); K(armN, 86, rot=(-30, 10, 0))
K(head, 86, rot=(8, 0, 0)); K(head, 92, rot=(0, 0, 0))
blink(90)

# ---- 3. big yawn + stretch
K(armP, 96, rot=(-30, -10, 0)); K(armN, 96, rot=(-30, 10, 0))
K(armP, 106, rot=(-165, 30, 0)); K(armN, 106, rot=(-165, -30, 0))
K(armP, 118, rot=(-172, 42, 0)); K(armN, 118, rot=(-172, -42, 0))
K(torso, 96, scl=1.0, rot=(0, 0, 0)); K(torso, 108, scl=(0.94, 0.94, 1.12), rot=(-10, 0, 0)); K(torso, 118, scl=(0.93, 0.93, 1.14), rot=(-13, 0, 0))
K(head, 96, rot=(0, 0, 0)); K(head, 108, rot=(-20, 0, 6)); K(head, 118, rot=(-24, 0, -6))
K(mouthO, 98, scl=0.001); K(mouthO, 107, scl=1.25); K(mouthO, 117, scl=1.3); K(mouthO, 124, scl=0.001)
eyes(99, 'open'); eyes(102, 'happy'); eyes(123, 'happy'); eyes(127, 'open')
K(torso, 125, scl=(1.06, 1.06, 0.93), rot=(4, 0, 0)); K(torso, 131, scl=1.0, rot=(0, 0, 0))
K(armP, 127, rot=ARM_REST_P); K(armN, 127, rot=ARM_REST_N)
K(head, 127, rot=(4, 0, 0)); K(head, 132, rot=(0, 0, 0))
for i, f in enumerate(range(126, 134, 2)):        # happy little shake
    K(root, f, rot=(0, 0, BED_YAW + (5 if i % 2 else -5)))
K(root, 134, loc=BEDP, rot=(0, 0, BED_YAW), scl=1.0)

# ---- 4. hop out of bed and bounce over to the desk
WALK_YAW = 78.0
hops = [BEDP, Vector((-1.75, -0.35, 0)), Vector((-0.75, -0.45, 0)), Vector((0.3, -0.35, 0))]
f = 134
K(pelvis, f, loc=(0, 0, 0.70))
K(root, f + 3, scl=(1.1, 1.1, 0.84))              # anticipation
K(armP, f + 3, rot=(20, -20, 0)); K(armN, f + 3, rot=(20, 20, 0))
f += 3
for i in range(3):
    a, bq = hops[i], hops[i + 1]
    hgt = 0.75 if i == 0 else 0.5
    yaw0 = BED_YAW if i == 0 else WALK_YAW
    K(root, f, loc=a, rot=(0, 0, yaw0))
    K(root, f + 2, scl=(0.9, 0.9, 1.16))
    K(root, f + 6, loc=(a + bq) / 2 + Vector((0, 0, hgt)), rot=(0, 0, WALK_YAW), scl=(0.95, 0.95, 1.08))
    K(root, f + 11, loc=bq, rot=(0, 0, WALK_YAW), scl=(0.95, 0.95, 1.1))
    K(root, f + 13, scl=(1.12, 1.12, 0.82))
    K(pelvis, f + 6, loc=(0, 0, HIP_Z + 0.1)); K(pelvis, f + 11, loc=(0, 0, HIP_Z))
    sw = 1 if i % 2 == 0 else -1
    K(legP, f + 5, rot=(-35 * sw - 10, 0, 6)); K(legN, f + 5, rot=(35 * sw - 10, 0, -6))
    K(legP, f + 11, rot=(0, 0, 0)); K(legN, f + 11, rot=(0, 0, 0))
    K(armP, f + 5, rot=(-80 + 40 * sw, -55, 0)); K(armN, f + 5, rot=(-80 - 40 * sw, 55, 0))
    K(armP, f + 12, rot=(10, -18, 0)); K(armN, f + 12, rot=(10, 18, 0))
    K(head, f + 5, rot=(-8, 0, 5 * sw)); K(head, f + 12, rot=(8, 0, 0))
    K(tail, f, rot=(0, 0, 22 * sw)); K(tail, f + 6, rot=(-10, 0, -22 * sw)); K(tail, f + 12, rot=(8, 0, 12 * sw))
    f += 13
# f == 176: final spinning hop, plop down on the cushion
SIT_Z = 0.50
K(root, f, loc=hops[3], rot=(0, 0, WALK_YAW))
K(root, f + 2, scl=(0.9, 0.9, 1.16))
K(root, f + 7, loc=(hops[3] + SEAT) / 2 + Vector((0, 0, 0.6)), rot=(0, 0, 10), scl=(0.95, 0.95, 1.08))
K(root, f + 13, loc=SEAT, rot=(0, 0, SEAT_YAW), scl=(0.96, 0.96, 1.08))
K(root, f + 15, scl=(1.14, 1.14, 0.80)); K(root, f + 19, scl=(0.97, 0.97, 1.05)); K(root, f + 23, scl=1.0)
K(pelvis, f, loc=(0, 0, HIP_Z)); K(pelvis, f + 7, loc=(0, 0, HIP_Z + 0.2)); K(pelvis, f + 13, loc=(0, 0, SIT_Z))
K(legP, f + 2, rot=(0, 0, 0)); K(legN, f + 2, rot=(0, 0, 0))
K(legP, f + 8, rot=(-70, 0, 20)); K(legN, f + 8, rot=(-70, 0, -20))
K(legP, f + 13, rot=(-88, 0, 44)); K(legN, f + 13, rot=(-88, 0, -44))
K(legP, f + 15, rot=(-80, 0, 46)); K(legN, f + 15, rot=(-80, 0, -46)); K(legP, f + 19, rot=(-88, 0, 44)); K(legN, f + 19, rot=(-88, 0, -44))
K(armP, f + 7, rot=(-150, 45, 0)); K(armN, f + 7, rot=(-150, -45, 0))
K(armP, f + 15, rot=(-20, -30, 0)); K(armN, f + 15, rot=(-20, 30, 0))
K(head, f + 7, rot=(-10, 0, 0)); K(head, f + 15, rot=(12, 0, 0)); K(head, f + 21, rot=(0, 0, 0))
K(tail, f + 7, rot=(-15, 0, 25)); K(tail, f + 15, rot=(20, 0, -15)); K(tail, f + 23, rot=(0, 0, 0))
K(seat_c, f + 12, scl=1.0); K(seat_c, f + 15, scl=(1.08, 1.08, 0.72)); K(seat_c, f + 20, scl=(1.03, 1.03, 0.86))
K(torso, f + 13, rot=(0, 0, 0), scl=1.0)
blink(f + 18)
SITF = f + 15                                     # 191

# ---- 5. laptop pops open, to-do board pops up
K(lid, 1, rot=(0, 0, 0)); K(lid, SITF + 4, rot=(0, 0, 0)); K(lid, SITF + 11, rot=(118, 0, 0)); K(lid, SITF + 14, rot=(102, 0, 0)); K(lid, SITF + 17, rot=(108, 0, 0))
SCREEN_E.default_value = 0.0; SCREEN_E.keyframe_insert('default_value', frame=SITF + 8)
SCREEN_E.default_value = 9.0; SCREEN_E.keyframe_insert('default_value', frame=SITF + 13)
K(armP, SITF + 4, rot=(-20, -30, 0)); K(armN, SITF + 4, rot=(-20, 30, 0))
K(armN, SITF + 9, rot=(-75, 10, 0)); K(armN, SITF + 14, rot=(-58, 14, 0))       # flips the lid open
K(board, 1, scl=0.001); pop(board, SITF + 16, 1.0, dur=10)
K(head, SITF + 16, rot=(0, 0, 0)); K(head, SITF + 22, rot=(-6, 0, -20)); K(head, SITF + 32, rot=(-6, 0, -20)); K(head, SITF + 38, rot=(6, 0, 0))
eyes(SITF + 24, 'open'); eyes(SITF + 26, 'happy'); eyes(SITF + 33, 'happy'); eyes(SITF + 35, 'open')
K(CHECKS[0], 1, scl=0.001); pop(CHECKS[0], SITF + 24, 1.0, dur=7)
for c in CHECKS[1:]:
    K(c, 1, scl=0.001)

# ---- 6. gets to work: typing, a checked box, a cheer, back to typing
T0 = SITF + 38                                    # 229
K(torso, T0 - 8, rot=(0, 0, 0)); K(torso, T0, rot=(17, 0, 0))
TYP_P, TYP_N = (-50, 20, 0), (-50, -20, 0)
K(armP, T0 - 6, rot=(-20, -30, 0)); K(armN, T0 - 6, rot=(-58, 14, 0))
def typing(f0, f1):
    i = 0
    for f in range(f0, f1, 3):
        up = i % 2
        K(armP, f, rot=(TYP_P[0] - (9 if up else 0), TYP_P[1], 0))
        K(armN, f, rot=(TYP_N[0] - (0 if up else 9), TYP_N[1], 0))
        K(head, f, rot=(8 + (2 if up else -1), 0, 3 if (i // 4) % 2 else -3))
        K(torso, f, rot=(17 + (1 if up else 0), 0, 0), scl=1.0)
        i += 1
CHK = T0 + 34                                     # check pops
typing(T0, CHK - 2)
pop(CHECKS[1], CHK, 1.0, dur=7)
# glance at the list, then cheer
K(head, CHK + 2, rot=(-4, 0, -18)); K(head, CHK + 9, rot=(-14, 0, 28)); K(head, CHK + 20, rot=(-12, 0, 24))
K(torso, CHK + 4, rot=(4, 0, 0), scl=(0.97, 0.97, 1.06)); K(torso, CHK + 20, rot=(2, 0, 0), scl=1.0)
K(armP, CHK + 2, rot=(-50, 20, 0)); K(armP, CHK + 8, rot=(-165, 48, 0)); K(armP, CHK + 11, rot=(-152, 40, 0)); K(armP, CHK + 14, rot=(-165, 48, 0)); K(armP, CHK + 20, rot=(-155, 42, 0))
K(armN, CHK + 2, rot=(-50, -20, 0)); K(armN, CHK + 8, rot=(-165, -48, 0)); K(armN, CHK + 11, rot=(-152, -40, 0)); K(armN, CHK + 14, rot=(-165, -48, 0)); K(armN, CHK + 20, rot=(-155, -42, 0))
K(mouthO, CHK + 2, scl=0.001); K(mouthO, CHK + 7, scl=0.95); K(mouthO, CHK + 20, scl=0.85); K(mouthO, CHK + 25, scl=0.001)
eyes(CHK + 3, 'open'); eyes(CHK + 5, 'happy'); eyes(CHK + 22, 'happy'); eyes(CHK + 25, 'open')
K(root, CHK + 2, loc=SEAT, scl=1.0); K(root, CHK + 6, loc=SEAT + Vector((0, 0, 0.14)), scl=(0.96, 0.96, 1.08)); K(root, CHK + 10, loc=SEAT, scl=(1.06, 1.06, 0.92)); K(root, CHK + 14, loc=SEAT, scl=1.0)
for i, f in enumerate(range(SITF + 23, END + 4, 9)):
    K(tail, f, rot=(0, 0, 16 if i % 2 else -16))
K(head, CHK + 25, rot=(-4, 0, 8)); K(head, CHK + 30, rot=(8, 0, 0))
K(torso, CHK + 28, rot=(17, 0, 0))
K(armP, CHK + 28, rot=TYP_P); K(armN, CHK + 28, rot=TYP_N)
typing(CHK + 30, END + 3)

# sparkles
def sparkle(f0, center, n=5, spread=0.9, seed=0):
    rng = np.random.default_rng(seed)
    for i in range(n):
        s = star(f'sparkle_{f0}_{i}', r=float(rng.uniform(0.13, 0.24)))
        a = 2 * math.pi * i / n + rng.uniform(-0.3, 0.3)
        d = Vector((math.cos(a) * spread, -0.25, math.sin(a) * spread * 0.8))
        ff = f0 + int(rng.integers(0, 4))
        K(s, 1, loc=center, scl=0.001)
        K(s, ff, loc=Vector(center) + d * 0.5, scl=0.001, rot=(90, 0, 0))
        K(s, ff + 4, loc=Vector(center) + d * 0.85, scl=1.3, rot=(90, 40, 0))
        K(s, ff + 9, loc=Vector(center) + d, scl=0.9, rot=(90, 80, 0))
        K(s, ff + 14, loc=Vector(center) + d * 1.08, scl=0.001, rot=(90, 130, 0))
sparkle(SITF + 19, (-0.75, 1.3, 1.35), n=6, spread=1.25, seed=1)
sparkle(CHK + 5, SEAT + Vector((0, 0, 1.75)), n=6, spread=1.05, seed=2)
sparkle(102, BEDP + Vector((0, 0, 2.0)), n=4, spread=1.0, seed=3)

# lighting: dawn -> bright morning
def kval(sock_or_obj, prop, f, v):
    setattr(sock_or_obj, prop, v); sock_or_obj.keyframe_insert(prop, frame=f)
for f, w, s, k in ((1, 0.22, 1.2, 0.35), (54, 0.35, 4.0, 0.6), (110, 0.5, 7.0, 0.9), (END, 0.5, 7.0, 0.9)):
    kval(WORLD_E, 'default_value', f, w)
    kval(sun_d, 'energy', f, s)
    kval(key.data, 'energy', f, 2600 * k)
    kval(rim_l.data, 'energy', f, 900 * k)

# camera
K(cam, 1, loc=(-1.7, -6.0, 2.9)); K(aim, 1, loc=BED + Vector((0.1, 0, 1.0)))
K(cam, 60, loc=(-1.9, -5.7, 2.7)); K(aim, 60, loc=BED + Vector((0.15, 0, 1.1)))
K(cam, 96, loc=(-2.0, -6.0, 2.4)); K(aim, 96, loc=BED + Vector((0.3, 0, 1.45)))
K(cam, 134, loc=(-2.0, -6.4, 2.3)); K(aim, 134, loc=BED + Vector((0.4, 0, 1.45)))
K(cam, 162, loc=(-0.6, -7.2, 2.1)); K(aim, 162, loc=(-0.9, 0.3, 1.3))
K(cam, 196, loc=(0.4, -7.0, 2.0)); K(aim, 196, loc=(0.45, 0.6, 1.15))
K(cam, END, loc=(0.6, -6.3, 1.9)); K(aim, END, loc=(0.35, 0.6, 1.12))

# ----------------------------------------------------------------------------- hero / turnaround mode
if arg('--hero'):
    yaw, cx, cy_, cz = [float(v) for v in arg('--hero', 4)]
    scene.frame_set(END)
    for o in bpy.data.objects:
        o.animation_data_clear()
    for e in ALL_J:
        e.rotation_euler = (0, 0, 0); e.scale = (1, 1, 1)
        e.location = REST[e.name] - (REST[e.parent.name] if e.parent else Vector((0, 0, 0)))
    armP.rotation_euler = [R(a) for a in ARM_REST_P]; armN.rotation_euler = [R(a) for a in ARM_REST_N]
    for eo, ec in EYES.values():
        eo.scale = (1, 1, 1); ec.scale = (0.001,) * 3
    mouthO.scale = (0.001,) * 3
    HP = Vector((-1.2, -1.6, 0))
    root.location = HP; root.rotation_euler = (0, 0, R(yaw))
    cam.location = HP + Vector((cx, cy_, cz)); aim.location = HP + Vector((0, 0, 1.0))
    cam_d.lens = 60; cam_d.dof.aperture_fstop = 8
# ----------------------------------------------------------------------------- render settings
scene.frame_start = 1; scene.frame_end = END
scene.render.fps = FPS
scene.render.engine = 'CYCLES'
cy = scene.cycles
cy.device = 'CPU'
cy.samples = int(arg('--samples', default=64))
cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.04
cy.use_denoising = True
cy.max_bounces = 5; cy.diffuse_bounces = 3; cy.glossy_bounces = 3; cy.transmission_bounces = 2
cy.caustics_reflective = False; cy.caustics_refractive = False
try:
    scene.render.use_persistent_data = True
except Exception:
    pass
res = arg('--res', 2, ['1280', '720'])
scene.render.resolution_x, scene.render.resolution_y = int(res[0]), int(res[1])
scene.render.resolution_percentage = 100
try:
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Punchy'
except Exception as e:
    print('look:', e)
scene.render.threads_mode = 'AUTO'

out = arg('--out')
if arg('--save'):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(arg('--save')))
if arg('--glb'):
    scene.frame_set(1)
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(arg('--glb')), export_apply=True, export_animations=True)
if arg('--still'):
    for fr in arg('--still').split(','):
        scene.frame_set(int(fr))
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = os.path.abspath(out % int(fr) if '%' in out else out)
        bpy.ops.render.render(write_still=True)
if arg('--anim'):
    a, bb = arg('--anim', 2)
    step = int(arg('--step', default=1))
    scene.render.image_settings.file_format = 'PNG'
    for fr in range(int(a), int(bb) + 1, step):
        fp = os.path.join(os.path.abspath(out), f'f{fr:04d}.png')
        if os.path.exists(fp):
            continue
        scene.frame_set(fr)
        scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)
print('DONE')
