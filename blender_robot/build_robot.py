"""Builds "Bolt", an animated robot figure, in Blender and saves bolt_robot.blend.

Usage:
    python build_robot.py                 # build + save .blend (and .glb)
    python build_robot.py --render DIR    # also render PNG frames into DIR
    python build_robot.py --still F.png   # also render one preview frame
"""
import math
import os
import sys

import bpy  # must load first
import bmesh
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 24
END = 120  # frame 121 == frame 1, so frames 1..120 loop seamlessly

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
col = scene.collection


# ---------------------------------------------------------------- materials
def material(name, color, rough=0.4, metallic=0.0, emission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    if emission:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emission
    m.diffuse_color = (*color, 1)  # viewport solid-mode colour
    return m


SHELL = material("Shell", (0.92, 0.92, 0.95), 0.25)
ACCENT = material("Accent", (1.0, 0.33, 0.05), 0.3)
JOINT = material("Joint", (0.08, 0.09, 0.11), 0.35, 0.8)
VISOR = material("Visor", (0.01, 0.012, 0.02), 0.08)
GLOW = material("Glow", (0.1, 0.9, 1.0), 0.3, emission=6.0)
FLOOR = material("Floor", (0.16, 0.2, 0.34), 0.6)


# ----------------------------------------------------------------- geometry
def add_box(bm, size, pos):
    m = Matrix.Translation(pos) @ Matrix.Diagonal((*size, 1))
    bmesh.ops.create_cube(bm, size=1.0, matrix=m)


def add_sphere(bm, radius, pos, scale=(1, 1, 1)):
    m = Matrix.Translation(pos) @ Matrix.Diagonal((*scale, 1))
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=radius, matrix=m)


def add_cyl(bm, radius, depth, pos, axis="Z", radius2=None):
    rot = {"Z": Matrix.Identity(4),
           "X": Matrix.Rotation(math.pi / 2, 4, "Y"),
           "Y": Matrix.Rotation(math.pi / 2, 4, "X")}[axis]
    bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=radius,
                          radius2=radius if radius2 is None else radius2,
                          depth=depth, matrix=Matrix.Translation(pos) @ rot)


def make(name, mat, origin, parent, build, bevel=0.0):
    """Create an object whose pivot sits at world-space `origin`.

    `build(bm, o)` adds geometry; `o(x, y, z)` converts world coords to local.
    """
    origin = Vector(origin)
    bm = bmesh.new()
    build(bm, lambda x, y, z: Vector((x, y, z)) - origin)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(mat)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.set_sharp_from_angle(angle=math.radians(40))
    ob = bpy.data.objects.new(name, mesh)
    col.objects.link(ob)
    if bevel:
        mod = ob.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 8
    ob["world_origin"] = origin
    if parent:
        ob.parent = parent
        ob.location = origin - Vector(parent["world_origin"])
    else:
        ob.location = origin
    return ob


root = bpy.data.objects.new("Bolt", None)  # master control, pivot on the floor
root.empty_display_type = "CIRCLE"
root.empty_display_size = 1.2
root.rotation_euler.x = 0  # circle lies flat when viewed from above
root["world_origin"] = Vector((0, 0, 0))
col.objects.link(root)

# Torso
body = make("Body", SHELL, (0, 0, 0.62), root,
            lambda bm, o: add_box(bm, (1.1, 0.8, 0.9), o(0, 0, 1.07)), bevel=0.2)
make("Belt", ACCENT, (0, 0, 0.62), body,
     lambda bm, o: add_box(bm, (1.13, 0.83, 0.12), o(0, 0, 0.86)), bevel=0.04)
make("ChestLight", GLOW, (0, -0.41, 1.22), body,
     lambda bm, o: add_cyl(bm, 0.13, 0.04, o(0, -0.41, 1.22), "Y"))
make("ChestRing", JOINT, (0, -0.4, 1.22), body,
     lambda bm, o: add_cyl(bm, 0.18, 0.03, o(0, -0.4, 1.22), "Y"))
make("Neck", JOINT, (0, 0, 1.5), body,
     lambda bm, o: add_cyl(bm, 0.17, 0.2, o(0, 0, 1.56)))

# Head
head = make("Head", SHELL, (0, 0, 1.6), body,
            lambda bm, o: add_box(bm, (1.35, 0.9, 0.85), o(0, 0, 2.05)), bevel=0.26)
make("Visor", VISOR, (0, -0.42, 2.05), head,
     lambda bm, o: add_box(bm, (1.02, 0.12, 0.5), o(0, -0.42, 2.05)), bevel=0.055)
eyes = []
for side, sx in (("L", -0.25), ("R", 0.25)):
    eyes.append(make(f"Eye.{side}", GLOW, (sx, -0.475, 2.07), head,
                     lambda bm, o, sx=sx: add_sphere(bm, 0.12, o(sx, -0.475, 2.07), (1, 0.3, 1.25))))
    make(f"Ear.{side}", ACCENT, (sx * 2.76, 0, 2.05), head,
         lambda bm, o, sx=sx: add_cyl(bm, 0.2, 0.1, o(sx * 2.76, 0, 2.05), "X"))


def antenna_geo(bm, o):
    add_cyl(bm, 0.025, 0.36, o(0, 0, 2.65))
    add_cyl(bm, 0.09, 0.05, o(0, 0, 2.49), radius2=0.06)


antenna = make("Antenna", JOINT, (0, 0, 2.47), head, antenna_geo)
make("AntennaTip", GLOW, (0, 0, 2.87), antenna,
     lambda bm, o: add_sphere(bm, 0.085, o(0, 0, 2.87)))

# Arms: pivot at the shoulder, forearm pivots at the elbow
arms, forearms = {}, {}
for side, sx in (("L", -1), ("R", 1)):
    x = sx * 0.69

    def upper(bm, o, x=x):
        add_sphere(bm, 0.16, o(x, 0, 1.36))
        add_cyl(bm, 0.085, 0.36, o(x, 0, 1.16))

    def lower(bm, o, x=x):
        add_sphere(bm, 0.115, o(x, 0, 0.97))
        add_cyl(bm, 0.075, 0.3, o(x, 0, 0.82))

    arm = make(f"Arm.{side}", JOINT, (x, 0, 1.36), body, upper)
    fore = make(f"Forearm.{side}", JOINT, (x, 0, 0.97), arm, lower)
    make(f"Hand.{side}", ACCENT, (x, 0, 0.62), fore,
         lambda bm, o, x=x: add_sphere(bm, 0.16, o(x, 0, 0.62)))
    arms[side], forearms[side] = arm, fore

# Legs
for side, sx in (("L", -0.28), ("R", 0.28)):
    leg = make(f"Leg.{side}", JOINT, (sx, 0, 0.64), root,
               lambda bm, o, sx=sx: add_cyl(bm, 0.1, 0.5, o(sx, 0, 0.42)))
    make(f"Foot.{side}", ACCENT, (sx, -0.06, 0.11), leg,
         lambda bm, o, sx=sx: add_box(bm, (0.38, 0.56, 0.22), o(sx, -0.06, 0.11)), bevel=0.09)

# ---------------------------------------------------------------- animation
scene.render.fps = FPS
scene.frame_start, scene.frame_end = 1, END
R = math.radians


def key(ob, path, frames):
    """frames: {frame: value}; value is a 3-tuple (or scalar applied to index)."""
    for f, v in sorted(frames.items()):
        setattr(ob, path, v)
        ob.keyframe_insert(path, frame=f)


LOOP = END + 1
REST_L, REST_R = (0, R(8), 0), (0, R(-8), 0)

# Right arm (screen-right, +X) lifts and waves; left arm swings along.
key(arms["R"], "rotation_euler", {
    1: REST_R, 12: (0, R(-150), 0), 46: (0, R(-150), 0), 54: (0, R(-25), 0),
    60: (0, R(-165), 0), 68: (0, R(-60), 0), 76: REST_R, LOOP: REST_R})
key(forearms["R"], "rotation_euler", {
    1: (0, 0, 0), 12: (0, R(35), 0), 18: (0, R(-30), 0), 24: (0, R(35), 0),
    30: (0, R(-30), 0), 36: (0, R(35), 0), 42: (0, R(-30), 0), 48: (0, 0, 0), LOOP: (0, 0, 0)})
key(arms["L"], "rotation_euler", {
    1: REST_L, 24: (0, R(14), 0), 46: REST_L, 54: (0, R(25), 0),
    60: (0, R(165), 0), 68: (0, R(60), 0), 76: REST_L, LOOP: REST_L})
key(forearms["L"], "rotation_euler", {
    1: (0, 0, 0), 46: (0, 0, 0), 54: (0, R(-30), 0), 60: (0, 0, 0), LOOP: (0, 0, 0)})

# Head tilts toward the wave, looks around after landing.
key(head, "rotation_euler", {
    1: (0, 0, 0), 12: (0, R(-12), R(8)), 44: (0, R(-12), R(8)), 52: (R(-10), 0, 0),
    60: (R(12), 0, 0), 70: (R(-8), 0, 0), 80: (0, 0, R(-25)), 92: (0, 0, R(-25)),
    100: (0, 0, R(20)), 110: (0, 0, R(20)), LOOP: (0, 0, 0)})

# Body sways during the wave.
key(body, "rotation_euler", {
    1: (0, 0, 0), 12: (0, R(-5), 0), 24: (0, R(-2), 0), 36: (0, R(-5), 0),
    48: (0, 0, 0), LOOP: (0, 0, 0)})

# Crouch -> jump with a full spin -> squashy landing.
key(root, "location", {
    1: (0, 0, 0), 54: (0, 0, 0), 60: (0, 0, 0.95), 62: (0, 0, 1.0), 68: (0, 0, 0),
    LOOP: (0, 0, 0)})
key(root, "scale", {
    1: (1, 1, 1), 46: (1, 1, 1), 53: (1.12, 1.12, 0.76), 57: (0.9, 0.9, 1.18),
    66: (0.94, 0.94, 1.1), 70: (1.16, 1.16, 0.72), 75: (0.96, 0.96, 1.06),
    80: (1, 1, 1), LOOP: (1, 1, 1)})
key(root, "rotation_euler", {
    1: (0, 0, 0), 54: (0, 0, 0), 68: (0, 0, R(360)), LOOP: (0, 0, R(360))})

# Antenna wobble (bigger after the landing) and eye blinks.
wob = {1: (0, 0, 0), LOOP: (0, 0, 0)}
for i, f in enumerate(range(7, 54, 6)):
    wob[f] = (0, R(7 if i % 2 else -7), 0)
for i, f in enumerate(range(70, 112, 4)):
    amp = 30 * (0.72 ** i)
    wob[f] = (R(amp * 0.4 * (1 if i % 2 else -1)), R(amp if i % 2 else -amp), 0)
key(antenna, "rotation_euler", wob)

blink = {1: (1, 1, 1), LOOP: (1, 1, 1)}
for f in (30, 84, 104):
    blink.update({f - 2: (1, 1, 1), f: (1, 1, 0.08), f + 2: (1, 1, 1)})
blink.update({66: (1, 1, 1), 70: (1.15, 1, 0.35), 76: (1, 1, 1)})  # landing squint
for eye in eyes:
    key(eye, "scale", blink)

# ----------------------------------------------------- stage, lights, camera
floor = make("Floor", FLOOR, (0, 0, 0), None,
             lambda bm, o: bmesh.ops.create_circle(bm, cap_ends=True, segments=96, radius=40))

world = bpy.data.worlds.new("World")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.16, 0.2, 0.34, 1)
bg.inputs["Strength"].default_value = 0.6
scene.world = world


def light(name, kind, loc, energy, color=(1, 1, 1), size=4.0, target=(0, 0, 1.4)):
    data = bpy.data.lights.new(name, kind)
    data.energy, data.color = energy, color
    if kind == "AREA":
        data.size = size
    ob = bpy.data.objects.new(name, data)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    col.objects.link(ob)


light("Key", "AREA", (4.5, -5, 6.5), 1800, (1.0, 0.95, 0.88))
light("Fill", "AREA", (-6, -4, 3), 500, (0.7, 0.82, 1.0), size=6)
light("Rim", "AREA", (-2.5, 6, 5), 1500, (1.0, 0.6, 0.35), size=3)

cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
cam.data.lens = 36
cam.location = (3.4, -8.6, 2.5)
cam.rotation_euler = (Vector((0, 0, 1.75)) - cam.location).to_track_quat("-Z", "Y").to_euler()
col.objects.link(cam)
scene.camera = cam

# ------------------------------------------------------------------ render
r = scene.render
r.resolution_x, r.resolution_y = 1280, 720
r.image_settings.file_format = "PNG"
try:
    scene.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    pass

engine = os.environ.get("BOLT_ENGINE", "CYCLES")
r.engine = engine
if engine == "CYCLES":
    scene.cycles.samples = int(os.environ.get("BOLT_SAMPLES", 48))
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
r.use_motion_blur = False
r.filepath = "//render/frame_"

blend_path = os.path.join(HERE, "bolt_robot.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path)
print("saved", blend_path)

args = sys.argv[1:]
if "--glb" in args:
    bpy.ops.export_scene.gltf(filepath=os.path.join(HERE, "bolt_robot.glb"),
                              export_format="GLB", export_animations=True)
if "--still" in args:
    out = args[args.index("--still") + 1]
    for f in (int(x) for x in os.environ.get("BOLT_FRAMES", "20").split(",")):
        scene.frame_set(f)
        r.filepath = out.replace(".png", f"_{f:03d}.png")
        bpy.ops.render.render(write_still=True)
if "--render" in args:
    out = args[args.index("--render") + 1]
    lo, hi = (int(x) for x in os.environ.get("BOLT_RANGE", f"1,{END}").split(","))
    for f in range(lo, hi + 1):
        scene.frame_set(f)
        r.filepath = os.path.join(out, f"frame_{f:04d}.png")
        if not os.path.exists(r.filepath):
            bpy.ops.render.render(write_still=True)
