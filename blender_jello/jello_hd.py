"""Jello the capybara (shark onesie) - high-fidelity static model.

Sculpted as signed-distance fields -> marching cubes (dense meshes), with real fur
fibres (Curves objects), seams, gathered cuffs, zipper, felt teeth and a studio set.

    python3 jello_hd.py -- --views hero --res 1600 2000 --samples 96 --out renders/
    python3 jello_hd.py -- --h 0.012 --hair 0 --views hero,front --res 640 800 --samples 24 --out prev/
    python3 jello_hd.py -- --turn 48 --res 1080 1350 --samples 48 --out turn/
    python3 jello_hd.py -- --hair 0.15 --save jello_hd.blend
"""
import bpy, bmesh, math, os, sys, time
import numpy as np
from mathutils import Vector, Euler
from skimage import measure

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def arg(name, n=1, default=None):
    if name in argv:
        i = argv.index(name)
        return argv[i + 1:i + 1 + n] if n > 1 else argv[i + 1]
    return default

H = float(arg('--h', default=0.006))          # voxel size of the main sculpt
HAIR = float(arg('--hair', default=1.0))      # fur density multiplier (0 = no fur)
T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:6.1f}s]', *a, flush=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
COL = scene.collection
R = math.radians
f32 = np.float32
RNG = np.random.default_rng(7)

# ============================================================================ sdf toolkit
def A(*v):
    return np.array(v, f32)

def rot(rx, ry, rz):
    return np.array(Euler((R(rx), R(ry), R(rz))).to_matrix(), f32)

def nrm(v):
    return np.sqrt((v * v).sum(-1))

def sph(p, c, r):
    return nrm(p - A(*c)) - r

def ell(p, c, r, rt=None):
    q = p - A(*c)
    if rt is not None:
        q = q @ rot(*rt)
    r = A(*r)
    k0 = nrm(q / r); k1 = nrm(q / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)

def cap(p, a, b, ra, rb=None):
    a = A(*a); b = A(*b); rb = ra if rb is None else rb
    ba = b - a; pa = p - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return nrm(pa - h[:, None] * ba) - (ra + (rb - ra) * h)

def rbox(p, c, b, r, rt=None):
    q = p - A(*c)
    if rt is not None:
        q = q @ rot(*rt)
    q = np.abs(q) - A(*b)
    return nrm(np.maximum(q, 0)) + np.minimum(q.max(-1), 0) - r

def basis(ax):
    ax = A(*ax); ax = ax / nrm(ax)
    u = np.cross(ax, A(0, 0, 1))
    if nrm(u) < 1e-3:
        u = A(1, 0, 0)
    u = u / nrm(u); v = np.cross(ax, u)
    return ax, u, v

def tor(p, c, ax, Rm, r, ribs=0, amp=0.0):
    ax, u, v = basis(ax)
    q = p - A(*c)
    z = q @ ax; x = q @ u; y = q @ v
    rad = np.sqrt(x * x + y * y)
    d = np.sqrt((rad - Rm) ** 2 + z * z) - r
    if ribs:
        d = d + amp * np.cos(ribs * np.arctan2(y, x))
    return d

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

def smax(a, b, k):
    return -smin(-a, -b, k)

def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

def vnoise(p, scale, seed=0.0):
    q = p.astype(np.float64) * scale + seed * 17.13
    i = np.floor(q); f = q - i; f = f * f * (3 - 2 * f)
    def hsh(dx, dy, dz):
        n = (i[:, 0] + dx) * 127.1 + (i[:, 1] + dy) * 311.7 + (i[:, 2] + dz) * 74.7
        s = np.sin(n) * 43758.5453
        return s - np.floor(s)
    def lerp(a, b, t):
        return a + (b - a) * t
    x00 = lerp(hsh(0, 0, 0), hsh(1, 0, 0), f[:, 0]); x10 = lerp(hsh(0, 1, 0), hsh(1, 1, 0), f[:, 0])
    x01 = lerp(hsh(0, 0, 1), hsh(1, 0, 1), f[:, 0]); x11 = lerp(hsh(0, 1, 1), hsh(1, 1, 1), f[:, 0])
    return lerp(lerp(x00, x10, f[:, 1]), lerp(x01, x11, f[:, 1]), f[:, 2]).astype(f32)

def fbm(p, scale, seed=0.0, oct=3):
    s = 0; a = 0.5
    for o in range(oct):
        s = s + a * vnoise(p, scale * 2 ** o, seed + o); a *= 0.5
    return s

def feval(f, P, chunk=1500000):
    out = np.empty(len(P), f32)
    for i in range(0, len(P), chunk):
        out[i:i + chunk] = f(P[i:i + chunk])
    return out

def grad(f, P, e=0.002):
    g = np.empty((len(P), 3), f32)
    for k in range(3):
        d = np.zeros(3, f32); d[k] = e
        g[:, k] = feval(f, P + d) - feval(f, P - d)
    return g / np.maximum(nrm(g), 1e-9)[:, None]

def sculpt(f, lo, hi, h):
    """Marching cubes of f over a box -> (V, T, N) with outward normals."""
    xs = np.arange(lo[0], hi[0] + h, h, dtype=f32); ys = np.arange(lo[1], hi[1] + h, h, dtype=f32)
    zs = np.arange(lo[2], hi[2] + h, h, dtype=f32)
    vol = np.empty((len(xs), len(ys), len(zs)), f32)
    cs = max(1, int(1500000 / (len(ys) * len(zs))))
    Y, Z = np.meshgrid(ys, zs, indexing='ij')
    for i in range(0, len(xs), cs):
        xx = xs[i:i + cs]
        P = np.empty((len(xx), len(ys), len(zs), 3), f32)
        P[..., 0] = xx[:, None, None]; P[..., 1] = Y[None]; P[..., 2] = Z[None]
        vol[i:i + cs] = f(P.reshape(-1, 3)).reshape(len(xx), len(ys), len(zs))
    V, T, _, _ = measure.marching_cubes(vol, 0.0, spacing=(h, h, h))
    V = V.astype(f32) + A(*lo); T = T.astype(np.int32)
    N = grad(f, V)
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    if (fn * N[T[:, 0]]).sum() < 0:
        T = T[:, ::-1].copy()
    return V, T, N

def ray(f, o, target, tmax=3.0):
    """Surface point + normal of f along the ray from an inside point o toward target."""
    o = A(*o); d = A(*target) - o; d = d / nrm(d)
    lo, hi = 0.0, tmax
    for _ in range(40):
        mid = (lo + hi) / 2
        if f((o + d * mid)[None, :])[0] < 0:
            lo = mid
        else:
            hi = mid
    p = o + d * lo
    n = grad(f, p[None, :])[0]
    return p, n

def seg_dist(P, a, b):
    a = A(*a); b = A(*b); ba = b - a; pa = P - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    return nrm(pa - h[:, None] * ba)

# ============================================================================ blender helpers
def hx(h):
    h = h.lstrip('#')
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return np.array([(x / 12.92) if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c], f32)

root = bpy.data.objects.new('jello', None); COL.objects.link(root)
def link(o, char=True):
    COL.objects.link(o)
    if char:
        o.parent = root
    return o

def new_mat(name):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree
    return m, nt, nt.nodes.get('Principled BSDF')

def plain(name, col, rough=0.5, metallic=0.0, coat=0.0, emit=0.0, sss=0.0, bump=0.0, bscale=300.0):
    m, nt, b = new_mat(name)
    c = tuple(hx(col)) + (1.0,) if isinstance(col, str) else col
    b.inputs['Base Color'].default_value = c
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metallic
    b.inputs['Coat Weight'].default_value = coat
    if emit:
        b.inputs['Emission Color'].default_value = c; b.inputs['Emission Strength'].default_value = emit
    if sss:
        b.inputs['Subsurface Weight'].default_value = sss
        b.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.3); b.inputs['Subsurface Scale'].default_value = 0.04
    if bump:
        add_bump(nt, b, bscale, bump)
    return m

def add_bump(nt, b, scale, strength, dist=0.01):
    tc = nt.nodes.new('ShaderNodeTexCoord'); n = nt.nodes.new('ShaderNodeTexNoise')
    n.inputs['Scale'].default_value = scale; n.inputs['Detail'].default_value = 4.0
    bp = nt.nodes.new('ShaderNodeBump'); bp.inputs['Strength'].default_value = strength; bp.inputs['Distance'].default_value = dist
    nt.links.new(tc.outputs['Object'], n.inputs['Vector']); nt.links.new(n.outputs[0], bp.inputs['Height'])
    nt.links.new(bp.outputs['Normal'], b.inputs['Normal'])

def attr_mat(name, rough=0.9, sheen=0.25, mult=1.0, sss=0.0, coat=0.0, bump=0.0, bscale=300.0):
    """Surface material that reads the per-vertex 'col' attribute."""
    m, nt, b = new_mat(name)
    at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = 'col'
    if mult != 1.0:
        mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'; mx.inputs[0].default_value = 1.0
        nt.links.new(at.outputs['Color'], mx.inputs[6]); mx.inputs[7].default_value = (mult, mult, mult, 1)
        nt.links.new(mx.outputs[2], b.inputs['Base Color'])
    else:
        nt.links.new(at.outputs['Color'], b.inputs['Base Color'])
    b.inputs['Roughness'].default_value = rough
    b.inputs['Sheen Weight'].default_value = sheen; b.inputs['Sheen Roughness'].default_value = 0.5
    b.inputs['Coat Weight'].default_value = coat; b.inputs['Coat Roughness'].default_value = 0.25
    if sss:
        b.inputs['Subsurface Weight'].default_value = sss
        b.inputs['Subsurface Radius'].default_value = (1.0, 0.4, 0.35); b.inputs['Subsurface Scale'].default_value = 0.05
    if bump:
        add_bump(nt, b, bscale, bump)
    return m

# fibres: colour from the per-strand attribute, darker toward the root (cheap self-shadowing depth)
M_FUR, nt, b = new_mat('fur_fibre')
at = nt.nodes.new('ShaderNodeAttribute'); at.attribute_name = 'col'
hi_ = nt.nodes.new('ShaderNodeHairInfo')
mr = nt.nodes.new('ShaderNodeMapRange'); mr.inputs['To Min'].default_value = 0.74; mr.inputs['To Max'].default_value = 1.10
nt.links.new(hi_.outputs['Intercept'], mr.inputs['Value'])
mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'; mx.blend_type = 'MULTIPLY'; mx.inputs[0].default_value = 1.0
nt.links.new(at.outputs['Color'], mx.inputs[6]); nt.links.new(mr.outputs[0], mx.inputs[7])
nt.links.new(mx.outputs[2], b.inputs['Base Color'])
b.inputs['Roughness'].default_value = 0.75
b.inputs['Sheen Weight'].default_value = 0.35; b.inputs['Sheen Roughness'].default_value = 0.4
b.inputs['Specular IOR Level'].default_value = 0.25

M_UNDER = attr_mat('under_fur', mult=0.8 if HAIR > 0 else 1.0, bump=0.0 if HAIR > 0 else 0.35, bscale=220)

def mesh_from(name, V, T, col=None, mat=None, char=True):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V)); me.vertices.foreach_set('co', np.ascontiguousarray(V, f32).ravel())
    me.loops.add(T.size); me.loops.foreach_set('vertex_index', np.ascontiguousarray(T, np.int32).ravel())
    k = T.shape[1]
    me.polygons.add(len(T)); me.polygons.foreach_set('loop_start', np.arange(len(T), dtype=np.int32) * k)
    me.polygons.foreach_set('loop_total', np.full(len(T), k, np.int32))
    me.polygons.foreach_set('use_smooth', np.ones(len(T), bool))
    me.update(calc_edges=True)
    if col is not None:
        ca = me.color_attributes.new('col', 'FLOAT_COLOR', 'POINT')
        rgba = np.ones((len(V), 4), f32); rgba[:, :3] = col
        ca.data.foreach_set('color', rgba.ravel())
    if mat is not None:
        me.materials.append(mat)
    return link(bpy.data.objects.new(name, me), char)

def vnormals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, T[:, k], fn)
    return N / np.maximum(nrm(N), 1e-12)[:, None]

def make_fur(name, V, T, N, col, ln, dens, rad=(0.0030, 0.0010), droop=0.35, jitter=0.30, clump=0.3, cell=0.022, seed=0):
    """Scatter fibres over a triangle mesh. col: (n,3), ln: (n,) fibre length per vertex."""
    if HAIR <= 0:
        return None
    rng = np.random.default_rng(seed)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * nrm(np.cross(b - a, c - a))
    wl = ln[T].mean(1)
    w = area * (wl > 0.0015)
    M = int(w.sum() * dens * HAIR)
    if M < 10:
        return None
    ti = rng.choice(len(T), M, p=w / w.sum())
    u = rng.random(M).astype(f32); v = rng.random(M).astype(f32)
    fl = u + v > 1; u[fl] = 1 - u[fl]; v[fl] = 1 - v[fl]
    bw = np.stack([1 - u - v, u, v], 1)[:, :, None]
    tri = T[ti]
    rootp = (V[tri] * bw).sum(1); n = (N[tri] * bw).sum(1); n /= np.maximum(nrm(n), 1e-9)[:, None]
    cc = (col[tri] * bw).sum(1); L = (ln[tri] * bw[:, :, 0]).sum(1)
    keep = L > 0.0015
    rootp, n, cc, L = rootp[keep], n[keep], cc[keep], L[keep]; M = len(L)
    L = L * rng.uniform(0.7, 1.12, M).astype(f32)
    g = A(0, 0, -1) - n * n[:, 2:3] * -1.0 * 0 - n * (n @ A(0, 0, -1))[:, None]       # gravity, tangent to the surface
    d = n + jitter * rng.normal(size=(M, 3)).astype(f32) + droop * g
    d /= nrm(d)[:, None]
    tip = rootp + d * L[:, None] + g * (0.35 * droop * L)[:, None]
    if clump > 0:
        warp = np.stack([vnoise(rootp, 0.7 / cell, 11.0 + k) for k in range(3)], 1) * (cell * 2.5)
        key = np.floor((rootp + warp) / cell).astype(np.int64)
        key = key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791
        _, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv).astype(f32)
        ctip = np.stack([np.bincount(inv, tip[:, k]) for k in range(3)], 1).astype(f32) / cnt[:, None]
        w_ = (clump * rng.uniform(0.5, 1.0, M)).astype(f32)[:, None]
        tip = tip * (1 - w_) + ctip[inv] * w_
    S = 4
    P = np.empty((M, S, 3), f32)
    for k in range(S):
        s = k / (S - 1)
        mid = rootp + d * (L * s)[:, None]                      # straight part
        P[:, k] = mid * (1 - s * s) + (rootp + (tip - rootp) * s) * (s * s)
    rd = np.empty((M, S), f32)
    for k in range(S):
        rd[:, k] = rad[0] + (rad[1] - rad[0]) * (k / (S - 1))
    cc = np.clip(cc * rng.uniform(0.9, 1.08, (M, 1)).astype(f32), 0, 1)
    cu = bpy.data.hair_curves.new(name)
    cu.add_curves([S] * M)
    cu.attributes['position'].data.foreach_set('vector', P.ravel())
    ra = cu.attributes.get('radius') or cu.attributes.new('radius', 'FLOAT', 'POINT')
    ra.data.foreach_set('value', rd.ravel())
    ca = cu.attributes.new('col', 'FLOAT_COLOR', 'CURVE')
    rgba = np.ones((M, 4), f32); rgba[:, :3] = cc
    ca.data.foreach_set('color', rgba.ravel())
    cu.materials.append(M_FUR)
    log(f'fur {name}: {M} fibres')
    return link(bpy.data.objects.new(name, cu))

def furry(name, V, T, N, col, ln, dens, **kw):
    mesh_from(name, V, T, col, M_UNDER)
    make_fur(name + '_fur', V, T, N, col, ln, dens, **kw)

def tube(name, pts, radius, mat, radii=None):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
        sp.points[i].radius = radii[i] if radii else 1.0
    sp.use_endpoint_u = True; sp.order_u = min(4, len(pts))
    cu.bevel_depth = radius; cu.bevel_resolution = 6; cu.resolution_u = 24; cu.use_fill_caps = True
    cu.materials.append(mat)
    return link(bpy.data.objects.new(name, cu))

def ball(name, c, r, mat, n=5):
    bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=n, radius=1.0)
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); me.shade_smooth(); me.materials.append(mat)
    o = bpy.data.objects.new(name, me); o.location = [float(x) for x in c]
    o.scale = (r,) * 3 if not hasattr(r, '__len__') else r
    return link(o)

def box(name, size, loc, mat, bevel=0.004, rt=(0, 0, 0), char=True):
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free(); me.materials.append(mat)
    o = bpy.data.objects.new(name, me); o.location = [float(x) for x in loc]; o.rotation_euler = [R(a) for a in rt]
    if bevel:
        bv = o.modifiers.new('bev', 'BEVEL'); bv.width = bevel; bv.segments = 4
    return link(o, char)

# ============================================================================ palette
C_BLUE = hx('#76A3EA'); C_NAVY = hx('#2B4578'); C_CUFF = hx('#6693DE'); C_CREAM = hx('#FFFCF5')
C_SKIN = hx('#E2A869'); C_SKIN_D = hx('#C48A57'); C_SKIN_L = hx('#EBB981'); C_EARIN = hx('#C87E66'); C_DARK = hx('#2E1A10')
C_PINK = hx('#F6A3B6'); C_BERRY = hx('#DE4F6C'); C_WAF = hx('#E9A548'); C_WAF_D = hx('#C9812B')

# ============================================================================ JELLO: head (skin)
def head_base(p):
    skull = ell(p, (0, 0.0, 1.44), (0.56, 0.50, 0.48))
    muz = rbox(p, (0, -0.46, 1.31), (0.115, 0.13, 0.065), 0.215, rt=(-6, 0, 0))
    jowl = ell(p, (0, -0.28, 1.14), (0.40, 0.36, 0.21))
    return smin(smin(skull, muz, 0.10), jowl, 0.10)

HO = (0, -0.15, 1.36)                                              # a point inside the head
NOS = [ray(head_base, HO, (sx * 0.185, -1.0, 1.50)) for sx in (1, -1)]
MOUTH = ray(head_base, HO, (0, -1.0, 1.02))
PHIL = [ray(head_base, HO, (0, -1.0, z))[0] for z in np.linspace(1.33, 1.03, 7)]
SMILE = [ray(head_base, HO, (x, -1.0, 1.035 + 0.05 * (abs(x) / 0.17) ** 1.6))[0] for x in np.linspace(-0.17, 0.17, 9)]
EYE = [ray(head_base, HO, (sx * 0.95, -0.47, 1.58)) for sx in (1, -1)]

def nostril_d(p):
    d = None
    for (c, n) in NOS:
        e = ell(p, c - n * 0.005, (0.052, 0.05, 0.036), rt=(0, 0, 0))
        d = e if d is None else np.minimum(d, e)
    return d

def crease_d(p):
    d = None
    for chain in (PHIL, SMILE):
        for a, b_ in zip(chain[:-1], chain[1:]):
            e = cap(p, a, b_, 0.010)
            d = e if d is None else np.minimum(d, e)
    return d

def mouth_d(p):
    c, n = MOUTH
    return ell(p, c - n * 0.01, (0.062, 0.06, 0.034))

def head_f(p):
    d = head_base(p)
    d = smax(d, -nostril_d(p), 0.02)
    d = smax(d, -crease_d(p), 0.012)
    d = smax(d, -mouth_d(p), 0.015)
    return d

# ============================================================================ JELLO: onesie
SH = {1: A(0.47, -0.02, 0.99), -1: A(-0.47, -0.02, 0.99)}          # shoulders
WR = {1: A(0.215, -0.55, 0.745), -1: A(-0.215, -0.55, 0.745)}      # wrists
ADIR = {s: (WR[s] - SH[s]) / nrm(WR[s] - SH[s]) for s in (1, -1)}
LEGX = 0.27

def body_d(p):
    return smin(ell(p, (0, 0.05, 0.62), (0.60, 0.56, 0.50)), ell(p, (0, 0.02, 0.96), (0.46, 0.42, 0.34)), 0.2)

def legs_d(p):
    d = None
    for sx in (1, -1):
        e = cap(p, (sx * LEGX, 0.0, 0.45), (sx * LEGX, -0.02, 0.19), 0.225, 0.185)
        th = np.arctan2(p[:, 1] + 0.02, p[:, 0] - sx * LEGX)
        e = e + 0.011 * np.sin(13 * th) * np.exp(-((p[:, 2] - 0.21) / 0.075) ** 2)   # gathers above the cuff
        d = e if d is None else np.minimum(d, e)
    return d

def sleeves_d(p):
    d = None
    for sx in (1, -1):
        a, w = SH[sx], WR[sx]
        e = cap(p, a, w, 0.215, 0.160)
        e = smin(e, sph(p, (a + w) / 2 + A(sx * 0.03, 0, -0.03), 0.205), 0.08)
        ax, u, v = basis(ADIR[sx])
        q = p - a; t = (q @ ax) / nrm(w - a)
        th = np.arctan2(q @ v, q @ u)
        e = e + 0.010 * np.sin(12 * th) * np.exp(-((t - 0.97) / 0.13) ** 2)           # gathers at the wrist
        e = e + 0.007 * np.sin(t * 34 + 1.5 * np.sin(th)) * np.exp(-((t - 0.52) / 0.2) ** 2)   # soft elbow folds
        d = e if d is None else np.minimum(d, e)
    return d

def hood_outer(p):
    d = smin(ell(p, (0, 0.06, 1.48), (0.71, 0.69, 0.65)), ell(p, (0, -0.36, 1.90), (0.45, 0.48, 0.21), rt=(-10, 0, 0)), 0.15)
    return smin(d, ell(p, (0, 0.12, 1.04), (0.56, 0.50, 0.28)), 0.12)

def cavity_d(p):
    return head_base(p) - 0.035

OPEN_C, OPEN_R = (0, -0.95, 1.27), (0.90, 1.0, 0.64)
def opening_d(p):
    return ell(p, OPEN_C, OPEN_R)

def hood_d(p):
    return smax(smax(hood_outer(p), -cavity_d(p), 0.03), -opening_d(p), 0.05)

def tail_d(p):
    d = cap(p, (0, 0.45, 0.45), (0, 0.98, 0.53), 0.25, 0.10)
    d = smin(d, ell(p, (0, 1.17, 0.80), (0.05, 0.15, 0.35), rt=(-36, 0, 0)), 0.07)
    return smin(d, ell(p, (0, 1.13, 0.30), (0.05, 0.13, 0.27), rt=(42, 0, 0)), 0.07)

def dorsal_d(p):
    return ell(p, (0, 0.76, 1.10), (0.055, 0.34, 0.19), rt=(42, 0, 0))

def garment_f(p):
    d = smin(body_d(p), legs_d(p), 0.06)
    d = smin(d, sleeves_d(p), 0.05)
    d = smin(d, hood_d(p), 0.05)
    d = smin(d, tail_d(p), 0.08)
    d = smin(d, dorsal_d(p), 0.05)
    return d + 0.012 * (fbm(p, 5.0, 3.0, 2) - 0.37)                # plush lumpiness

log('sculpting garment')
GV, GT, GN = sculpt(garment_f, (-0.86, -0.98, -0.02), (0.86, 1.52, 2.14), H)
log('garment verts', len(GV))

# --- regions, seams
bd, sl, lg, hd_, tl, do = [feval(f, GV) for f in (body_d, sleeves_d, legs_d, hood_d, tail_d, dorsal_d)]
other = np.minimum(np.minimum(sl, lg), np.minimum(hd_, np.minimum(tl, do)))
is_body = bd < other
is_hood = hd_ < np.minimum(np.minimum(bd, sl), np.minimum(tl, do)) + 0.0
x, y, z = GV[:, 0], GV[:, 1], GV[:, 2]
ee = np.sqrt((x / 0.375) ** 2 + ((z - 0.56) / 0.50) ** 2)
wz = 0.085 + 0.30 * np.clip((1.22 - z) / 0.66, 0, 1)
front = y < -0.05
belly = is_body & front & (((ee < 1) & (z < 0.56)) | ((np.abs(x) < wz) & (z >= 0.56)))
d_panel = np.where(z < 0.56, np.abs(ee - 1) * 0.42, np.abs(np.abs(x) - wz))
d_panel = np.where(is_body & front, d_panel, 9.0)
op = feval(opening_d, GV); cv = feval(cavity_d, GV)
lining = is_hood & ((op < 0.085) | (cv < 0.03))
d_rim = np.where(is_hood & (cv > 0.03), np.abs(op - 0.085), 9.0)
HC = (0, 0.06, 1.48)
GILL = []
for sx in (1, -1):
    for i in range(3):
        a_, _ = ray(hood_outer, HC, (sx * 1.0, -0.20 + 0.13 * i, 1.70))
        b_, _ = ray(hood_outer, HC, (sx * 1.0, -0.13 + 0.13 * i, 1.48))
        GILL.append((a_, b_))
d_gill = np.min([seg_dist(GV, a_, b_) for a_, b_ in GILL], 0)
gill = d_gill < 0.017
d_seam = np.full(len(GV), 9.0, f32)
d_seam = np.minimum(d_seam, np.where((is_hood & (cv > 0.03) & ((z > 1.5) | (y > 0.3))) | (is_body & (y > 0.25)) | (tl < 0.02) | (do < 0.02), np.abs(x), 9.0))
for sx in (1, -1):                                                 # armholes
    d_seam = np.minimum(d_seam, np.where((sl < 0.04) | (bd < 0.04), np.abs(nrm(GV - SH[sx]) - 0.262), 9.0) + np.where(x * sx > 0.2, 0, 9.0))
d_seam = np.minimum(d_seam, np.minimum(d_panel, d_rim))
d_seam = np.minimum(d_seam, np.where(is_body & (z < 0.42) & (np.abs(x) < 0.12), np.abs(z - 0.30) * 3, 9.0))   # crotch join
groove = np.exp(-(d_seam / 0.011) ** 2) + 0.8 * np.exp(-(d_gill / 0.012) ** 2)
zip_tape = is_body & front & (np.abs(x) < 0.034)
GV = GV - GN * (0.011 * groove)[:, None]

gcol = np.tile(C_BLUE, (len(GV), 1))
gcol[belly | lining | zip_tape] = C_CREAM
gcol[gill] = C_NAVY
shade = 1.0 - 0.10 * sstep(0.45, 0.75, fbm(GV, 9.0, 5.0, 2))       # subtle pile variation
gcol = gcol * shade[:, None]
glen = 0.030 * (1 - 0.5 * np.clip(groove, 0, 1))
glen[gill] *= 0.5
glen[zip_tape] = 0.006
furry('onesie', GV, GT, GN, gcol, glen.astype(f32), 240000, seed=1)

# ============================================================================ skin parts
log('sculpting head')
HV, HT, HN = sculpt(head_f, (-0.62, -0.88, 0.90), (0.62, 0.56, 1.96), min(H, 0.0045) if H < 0.01 else H)
x, y, z = HV[:, 0], HV[:, 1], HV[:, 2]
nd = feval(nostril_d, HV); cd = feval(crease_d, HV); md = feval(mouth_d, HV)
hcol = np.tile(C_SKIN, (len(HV), 1))
t = (sstep(-0.42, -0.80, y) * sstep(1.20, 1.40, z))[:, None]       # warmer, darker bridge of the nose
hcol = hcol * (1 - 0.75 * t) + C_SKIN_D * (0.75 * t)
t = sstep(1.16, 0.98, z)[:, None]                                  # lighter chin
hcol = hcol * (1 - 0.6 * t) + C_SKIN_L * (0.6 * t)
dark = np.clip(sstep(0.020, -0.004, nd) + 0.8 * sstep(0.016, 0.002, cd) + sstep(0.02, -0.005, md), 0, 1)[:, None]
hcol = hcol * (1 - dark) + C_DARK * dark
hlen = 0.017 * (1 - dark[:, 0])
hlen = hlen * (0.55 + 0.45 * sstep(-0.80, -0.55, y))               # shorter velvet on the nose
for (c, n) in EYE:
    hlen = hlen * sstep(0.085, 0.13, nrm(HV - c))
furry('head', HV, HT, HN, hcol, hlen.astype(f32), 520000, rad=(0.0024, 0.0008), droop=0.5, jitter=0.25, clump=0.0, seed=2)

# eyes, lashes, brows
M_EYE = plain('eye', '#070504', rough=0.12, coat=0.6)
M_EYE.node_tree.nodes['Principled BSDF'].inputs['Specular IOR Level'].default_value = 0.25
M_EYE.node_tree.nodes['Principled BSDF'].inputs['Coat Roughness'].default_value = 0.06
M_GLINT = plain('glint', '#FFFFFF', rough=0.2, emit=2.5)
M_LASH = plain('lash', '#1C100A', rough=0.6)
for (c, n), sx in zip(EYE, (1, -1)):
    c = Vector(c); n = Vector(n)
    ec = c - n * 0.036
    ball('eyeball', ec, 0.092, M_EYE)
    up = (Vector((0, 0, 1)) - n * n.z).normalized()
    side = n.cross(up) * sx                                         # toward the back of the head
    ball('glint', ec + (n + up * 0.42 - side * 0.38).normalized() * 0.084, 0.024, M_GLINT, n=3)
    ball('glint2', ec + (n - up * 0.35 + side * 0.30).normalized() * 0.088, 0.010, M_GLINT, n=3)
    for i, a in enumerate((2, 26, 50)):
        dv = side * math.cos(R(a)) + up * math.sin(R(a))
        base = ec + (n * 0.55 + dv).normalized() * 0.090
        mid = base + dv * 0.022 + n * 0.004
        tip = base + (side * math.cos(R(a + 20)) + up * math.sin(R(a + 20))) * 0.046 - n * 0.004
        tube(f'lash{i}', [base, mid, tip], 0.008, M_LASH, radii=[1.0, 0.75, 0.2])
    bp = []
    for a in np.linspace(48, 112, 6):
        dv = side * math.cos(R(a)) + up * math.sin(R(a))
        q, qn = ray(head_base, HO, tuple(c + dv * 0.20 + n * 0.25))
        bp.append(Vector(q) + Vector(qn) * 0.012)
    tube('brow', bp, 0.011, M_LASH, radii=[0.3, 0.9, 1, 1, 0.8, 0.3])

# ears (poke out through the hood)
log('sculpting ears, hands, feet')
def ear_parts(p, sx):
    rt = (-12, 0, sx * 38)
    c = A(sx * 0.53, 0.10, 2.00)
    fwd = rot(*rt) @ A(0, -1, 0)
    return ell(p, c, (0.16, 0.075, 0.16), rt), ell(p, c + fwd * 0.085, (0.10, 0.055, 0.10), rt)
def ears_f(p):
    d = None
    for sx in (1, -1):
        o, i = ear_parts(p, sx)
        e = smax(o, -i, 0.035)
        d = e if d is None else np.minimum(d, e)
    return d
EV, ET, EN = sculpt(ears_f, (-0.78, -0.15, 1.74), (0.78, 0.36, 2.24), min(H, 0.005) if H < 0.01 else H)
inner = np.minimum(feval(lambda p: ear_parts(p, 1)[1], EV), feval(lambda p: ear_parts(p, -1)[1], EV))
t = sstep(0.012, -0.004, inner)[:, None]
ecol = C_SKIN * (1 - t) + C_EARIN * t
furry('ears', EV, ET, EN, ecol, np.full(len(EV), 0.015, f32) * (1 - 0.4 * t[:, 0]), 520000, rad=(0.0024, 0.0008), clump=0.15, seed=3)

# hands (wrapped around the ice-cream cone)
HANDC = {s: WR[s] + ADIR[s] * 0.15 for s in (1, -1)}
def hands_f(p):
    d = None
    for sx in (1, -1):
        c = HANDC[sx]
        e = ell(p, c, (0.118, 0.135, 0.125))
        for dz in (-0.078, 0.0, 0.078):
            e = smin(e, ell(p, c + A(-sx * 0.075, -0.085, dz), (0.07, 0.058, 0.043)), 0.022)
        e = smin(e, ell(p, c + A(-sx * 0.02, -0.02, 0.115), (0.055, 0.06, 0.05)), 0.03)
        d = e if d is None else np.minimum(d, e)
    return d
V_, T_, N_ = sculpt(hands_f, (-0.36, -0.90, 0.50), (0.36, -0.45, 0.90), min(H, 0.004) if H < 0.01 else H)
furry('hands', V_, T_, N_, np.tile(C_SKIN, (len(V_), 1)), np.full(len(V_), 0.012, f32), 560000, rad=(0.0022, 0.0008), clump=0.12, seed=4)

def feet_f(p):
    d = None
    for sx in (1, -1):
        e = ell(p, (sx * LEGX, -0.09, 0.095), (0.175, 0.265, 0.10))
        for dx in (-0.105, 0.0, 0.105):
            e = smin(e, ell(p, (sx * LEGX + dx, -0.335, 0.078), (0.064, 0.085, 0.072)), 0.028)
        d = e if d is None else np.minimum(d, e)
    return smax(d, -(p[:, 2] - 0.006), 0.025)
V_, T_, N_ = sculpt(feet_f, (-0.50, -0.46, -0.01), (0.50, 0.22, 0.22), min(H, 0.004) if H < 0.01 else H)
fl = 0.012 * sstep(0.012, 0.03, V_[:, 2])
furry('feet', V_, T_, N_, np.tile(C_SKIN, (len(V_), 1)) * 0.97, fl.astype(f32), 560000, rad=(0.0022, 0.0008), clump=0.12, seed=5)

# ribbed knit cuffs
CUFFS = [(tuple(WR[s] + ADIR[s] * 0.04), tuple(ADIR[s]), 0.120, 0.052, 30) for s in (1, -1)] + \
        [((s * LEGX, -0.02, 0.165), (0, 0, 1), 0.160, 0.052, 38) for s in (1, -1)]
def cuffs_f(p):
    d = None
    for c, ax, Rm, r, ribs in CUFFS:
        e = tor(p, c, ax, Rm, r, ribs, 0.0065)
        d = e if d is None else np.minimum(d, e)
    return d
V_, T_, N_ = sculpt(cuffs_f, (-0.52, -0.80, 0.08), (0.52, 0.22, 0.96), min(H, 0.004) if H < 0.01 else H)
furry('cuffs', V_, T_, N_, np.tile(C_CUFF, (len(V_), 1)), np.full(len(V_), 0.007, f32), 700000, rad=(0.002, 0.0008), clump=0.0, jitter=0.5, seed=6)

# ============================================================================ hood details
# felt teeth along the upper rim of the hood opening
M_FELT = plain('felt_white', '#FBF7EE', rough=0.95, bump=0.5, bscale=500)
M_FELT.node_tree.nodes['Principled BSDF'].inputs['Sheen Weight'].default_value = 0.6
bm = bmesh.new(); bmesh.ops.create_cube(bm, size=2.0)
bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=7, use_grid_fill=True)
QV = np.array([v.co[:] for v in bm.verts], f32); QV /= nrm(QV)[:, None]
QF = np.array([[v.index for v in f.verts] for f in bm.faces], np.int32); bm.free()
TV, TF = [], []
oc = A(*OPEN_C); orr = A(*OPEN_R)
phis = np.linspace(R(33), R(147), 9)
for j, phi in enumerate(phis):
    def rim_pt(th):
        return oc + orr * A(math.sin(th) * math.cos(phi), -math.cos(th), math.sin(th) * math.sin(phi))
    ths = np.linspace(0.05, math.pi * 0.7, 300)
    vals = hood_outer(np.array([rim_pt(t_) for t_ in ths], f32)) + 0.05
    hit = np.nonzero(vals < 0)[0]
    if len(hit) == 0:
        continue
    lo_ = ths[hit[0]]
    p0 = rim_pt(lo_)
    radial = A(math.cos(phi), 0, math.sin(phi))
    dn = -radial + A(0, -0.10, 0); dn /= nrm(dn)
    tang = A(-math.sin(phi), 0, math.cos(phi))
    thick = np.cross(dn, tang)
    size = 1.0 - 0.28 * abs(j - 4) / 4
    Lt, Wt, Tt = 0.23 * size, 0.115 * size, 0.036
    tt = (QV[:, 2] + 1) / 2
    loc = np.stack([QV[:, 0] * Wt * (1 - 0.86 * tt), QV[:, 1] * Tt * (1 - 0.55 * tt), tt * Lt], 1)
    W = p0 - dn * 0.05 + loc[:, 0:1] * tang + loc[:, 1:2] * thick + loc[:, 2:3] * dn
    TF.append(QF + len(TV) * len(QV)); TV.append(W)
teeth = mesh_from('teeth', np.concatenate(TV), np.concatenate(TF), mat=M_FELT)
s_ = teeth.modifiers.new('sub', 'SUBSURF'); s_.levels = 1; s_.render_levels = 2

# shark eyes on the hood
for sx in (1, -1):
    p, n = ray(hood_outer, HC, (sx * 0.74, -0.62, 2.10))
    p = Vector(p); n = Vector(n)
    ball('shark_eye', p - n * 0.025, 0.078, M_EYE)
    ball('shark_glint', p - n * 0.025 + (n + Vector((0, -0.3, 0.45))).normalized() * 0.072, 0.02, M_GLINT, n=3)

# ============================================================================ zipper
M_ZIP = plain('zip_coil', '#EFE6D6', rough=0.45)
M_METAL = plain('zip_metal', '#C7CAD0', rough=0.22, metallic=1.0)
BO = (0, 0.0, 0.70)
zpts = []
for a in np.linspace(62, -66, 150):
    p, n = ray(body_d, BO, (0, -math.cos(R(a)) * 3, 0.70 + math.sin(R(a)) * 3))
    zpts.append((p, n))
bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1.0)
CV = np.array([v.co[:] for v in bm.verts], f32); CF = np.array([[v.index for v in f.verts] for f in bm.faces], np.int32); bm.free()
ZV, ZF = [], []
for i in range(1, len(zpts) - 1):
    p, n = zpts[i]
    tg = zpts[i + 1][0] - zpts[i - 1][0]; tg /= nrm(tg)
    sd = np.cross(tg, n); sgn = 1 if i % 2 else -1
    W = p + n * 0.004 + (CV[:, 0:1] * 0.017 + sgn * 0.0045) * sd + CV[:, 1:2] * 0.0075 * tg + CV[:, 2:3] * 0.009 * n
    ZF.append(CF + len(ZV) * 8); ZV.append(W)
zo = mesh_from('zip_teeth', np.concatenate(ZV), np.concatenate(ZF), mat=M_ZIP)
zo.data.polygons.foreach_set('use_smooth', np.zeros(len(zo.data.polygons), bool))
bv = zo.modifiers.new('bev', 'BEVEL'); bv.width = 0.0018; bv.segments = 2
sp_, sn_ = zpts[9]
ang = math.degrees(math.atan2(sn_[2], -sn_[1]))
box('zip_slider', (0.046, 0.022, 0.058), Vector(sp_) + Vector(sn_) * 0.016, M_METAL, bevel=0.006, rt=(ang, 0, 0))
box('zip_bridge', (0.016, 0.016, 0.03), Vector(sp_) + Vector(sn_) * 0.032 + Vector((0, 0, -0.004)), M_METAL, bevel=0.004, rt=(ang, 0, 0))
box('zip_pull', (0.040, 0.008, 0.105), Vector(sp_) + Vector(sn_) * 0.040 + Vector((0, -0.004, -0.060)), M_METAL, bevel=0.0035, rt=(ang - 14, 0, 6))

# ============================================================================ strawberry ice cream
log('sculpting ice cream')
P0 = A(0, -0.655, 0.47); P1 = A(0, -0.905, 0.905)
CAX, CU_, CV_ = basis(P1 - P0); CLEN = float(nrm(P1 - P0))
SC = P1 + CAX * 0.105                                              # scoop centre
def cone_parts(p):
    q = p - P0; t = q @ CAX
    th = np.arctan2(q @ CV_, q @ CU_)
    ridge = np.minimum(np.abs(np.sin(4.5 * th + t * 17)), np.abs(np.sin(4.5 * th - t * 17)))
    return t, sstep(0.34, 0.06, ridge)
def cone_f(p):
    t, rg = cone_parts(p)
    d = cap(p, P0, P1, 0.016, 0.130) - 0.0065 * rg
    d = smin(d, tor(p, P1 - CAX * 0.012, CAX, 0.128, 0.013), 0.01)   # rolled rim
    return smax(d, t - CLEN - 0.004, 0.006)
V_, T_, N_ = sculpt(cone_f, (-0.17, -1.06, 0.43), (0.17, -0.60, 0.99), 0.0028 if H < 0.01 else H)
_, rg = cone_parts(V_)
ccol = C_WAF_D * (1 - rg[:, None]) + C_WAF * rg[:, None]
ccol = ccol * (0.9 + 0.2 * vnoise(V_, 40.0, 2.0))[:, None]
mesh_from('waffle_cone', V_, T_, ccol, attr_mat('waffle', rough=0.62, sheen=0.0, bump=0.35, bscale=700))

def scoop_f(p):
    d = sph(p, SC, 0.168) + 0.030 * (fbm(p, 11.0, 1.0, 3) - 0.42)
    sk = tor(p, P1 + CAX * 0.012, CAX, 0.128, 0.050) + 0.035 * (fbm(p, 16.0, 4.0, 3) - 0.45)
    d = smin(d, sk, 0.035)
    return d + 0.012 * sstep(0.62, 0.8, vnoise(p, 30.0, 9.0))     # little pits
V_, T_, N_ = sculpt(scoop_f, (-0.24, -1.20, 0.86), (0.24, -0.72, 1.24), 0.0028 if H < 0.01 else H)
bits = sstep(0.60, 0.74, vnoise(V_, 30.0, 9.0))[:, None]
scol = (C_PINK * (0.93 + 0.14 * fbm(V_, 14.0, 6.0, 2))[:, None]) * (1 - bits) + C_BERRY * bits
mesh_from('ice_cream', V_, T_, scol, attr_mat('strawberry_icecream', rough=0.42, sheen=0.0, sss=0.45, coat=0.25, bump=0.5, bscale=260))

# tongue: licks the scoop
mc, mn = MOUTH
TG0 = mc - mn * 0.03; TG1 = SC + A(0, 0.075, 0.150)
def tongue_f(p):
    m_ = (TG0 + TG1) / 2
    ax, u, v = basis(TG1 - TG0)
    q = p - m_
    loc = np.stack([q @ u, q @ v, q @ ax], 1)
    loc[:, 1] = loc[:, 1] + 6.0 * loc[:, 2] ** 2 * (1 if v[2] < 0 else -1)    # gentle droop
    r = A(0.050, 0.019, float(nrm(TG1 - TG0)) / 2 + 0.035)
    k0 = nrm(loc / r); k1 = nrm(loc / (r * r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)
V_, T_, N_ = sculpt(tongue_f, tuple(np.minimum(TG0, TG1) - 0.09), tuple(np.maximum(TG0, TG1) + 0.09), 0.0025 if H < 0.01 else H)
mesh_from('tongue', V_, T_, None, plain('tongue', '#EE7F94', rough=0.32, coat=0.4, sss=0.5, bump=0.25, bscale=900))

# ============================================================================ studio
log('studio')
prof = [(-14.0, 0.0)] + [(4.0 + 3.0 * math.sin(a), 3.0 - 3.0 * math.cos(a)) for a in np.linspace(0, math.pi / 2, 24)] + [(7.0, 16.0)]
SV = np.array([(sx * 40.0, yy, zz) for (yy, zz) in prof for sx in (-1, 1)], f32)
SF = np.array([[2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2] for i in range(len(prof) - 1)], np.int32)
M_CYC, nt, b = new_mat('backdrop'); b.inputs['Base Color'].default_value = tuple(hx('#EFD9CC')) + (1,); b.inputs['Roughness'].default_value = 0.8
mesh_from('backdrop', SV, SF, mat=M_CYC, char=False)

world = bpy.data.worlds.new('world'); scene.world = world; world.use_nodes = True
bg = world.node_tree.nodes['Background']; bg.inputs['Color'].default_value = tuple(hx('#F2E6E0')) + (1,); bg.inputs['Strength'].default_value = 0.12

def area(name, loc, target, size, energy, col='#FFFFFF'):
    d = bpy.data.lights.new(name, 'AREA'); d.size = size; d.energy = energy; d.color = tuple(hx(col))
    o = bpy.data.objects.new(name, d); COL.objects.link(o); o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    return o
area('key', (-4.2, -5.2, 5.2), (0, -0.1, 1.1), 5.0, 850, '#FFF3E6')
area('fill', (5.0, -4.5, 2.2), (0, -0.1, 1.0), 6.0, 130, '#E6EEFF')
area('rim_r', (4.2, 4.6, 4.6), (0, 0.1, 1.2), 3.0, 900, '#FFF6EC')
area('rim_l', (-5.0, 3.8, 3.0), (0, 0.1, 1.1), 3.0, 520, '#DCE8FF')
area('top', (0, 0.5, 7.0), (0, 0, 1.0), 6.0, 160)

cam_d = bpy.data.cameras.new('cam'); cam_d.sensor_fit = 'VERTICAL'; cam_d.sensor_height = 36
cam = bpy.data.objects.new('cam', cam_d); COL.objects.link(cam); scene.camera = cam
aim = bpy.data.objects.new('aim', None); COL.objects.link(aim)
tr = cam.constraints.new('TRACK_TO'); tr.target = aim; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'
cam_d.dof.use_dof = True; cam_d.dof.focus_object = aim

# name: (aim point, azimuth deg (0 = straight on, + = toward her left), elevation deg, distance, lens, f-stop)
VIEWS = {
    'hero':   ((0.0, 0.10, 1.06), -38, 7, 8.6, 85, 8.0),
    'front':  ((0.0, 0.10, 1.06), 0, 5, 8.6, 85, 8.0),
    'side':   ((0.0, 0.20, 1.06), -90, 5, 9.2, 85, 8.0),
    'back':   ((0.0, 0.20, 1.06), -145, 10, 9.2, 85, 8.0),
    'face':   ((0.0, -0.45, 1.42), -30, 3, 4.6, 110, 5.6),
    'treat':  ((0.0, -0.72, 0.90), 28, 8, 3.6, 110, 5.6),
    'hood':   ((0.12, -0.35, 1.80), -24, -4, 3.8, 110, 5.6),
    'fabric': ((0.45, -0.25, 0.80), -55, 4, 2.6, 110, 5.6),
}
def set_view(name):
    a, az, el, dist, lens, fs = VIEWS[name]
    root.rotation_euler = (0, 0, R(-az))
    aim.location = Euler((0, 0, R(-az))).to_matrix() @ Vector(a)
    cam.location = Vector(aim.location) + Vector((0, -math.cos(R(el)), math.sin(R(el)))) * dist
    cam_d.lens = lens; cam_d.dof.aperture_fstop = fs

# ============================================================================ render
scene.render.engine = 'CYCLES'
cy = scene.cycles
cy.device = 'CPU'
cy.samples = int(arg('--samples', default=96))
cy.use_adaptive_sampling = True; cy.adaptive_threshold = 0.02
cy.use_denoising = True
cy.max_bounces = 6; cy.diffuse_bounces = 4; cy.glossy_bounces = 3; cy.transmission_bounces = 2
cy.caustics_reflective = False; cy.caustics_refractive = False
scene.cycles_curves.shape = 'RIBBONS'; scene.cycles_curves.subdivisions = 2
scene.render.use_persistent_data = True
res = arg('--res', 2, ['1600', '2000'])
scene.render.resolution_x, scene.render.resolution_y = int(res[0]), int(res[1])
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = 'AgX'
try:
    scene.view_settings.look = 'AgX - Medium High Contrast'
except Exception as e:
    print('look:', e)
scene.render.image_settings.file_format = 'PNG'
out = os.path.abspath(arg('--out', default='.'))

if arg('--save'):
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(arg('--save')), compress=True)
    log('saved')
if arg('--views'):
    for name in arg('--views').split(','):
        fp = os.path.join(out, f'jello_{name}.png')
        if os.path.exists(fp):
            continue
        set_view(name)
        scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        log('rendered', name)
if arg('--turn'):
    n = int(arg('--turn'))
    set_view('front')
    aim.location = (0, 0, 1.06)
    cam.location = Vector(aim.location) + Vector((0, -math.cos(R(6)), math.sin(R(6)))) * 9.4
    for i in range(n):
        fp = os.path.join(out, f't{i:04d}.png')
        if os.path.exists(fp):
            continue
        root.rotation_euler = (0, 0, R(38 + 360.0 * i / n))
        scene.render.filepath = fp
        bpy.ops.render.render(write_still=True)
        log('turn', i)
log('DONE')
