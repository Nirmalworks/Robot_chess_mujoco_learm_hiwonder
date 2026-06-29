import mujoco
import numpy as np

ARM_PREFIX = "arm_"
ARM_MOUNT_POS = (-0.03, 0.0, 0.0)   # base_link position on the tabletop
SQUARE = 0.018                       # must match the board size in table_scene.xml (0.144 m / 8)
BOARD_SURFACE_Z = 0.0046             # local z of the board's top face, see table_scene.xml board_slab

CONTROLLED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll", "grip_left"]
MIMIC = [
    # (dependent_joint, master_joint, multiplier)
    ("grip_right", "grip_left", -1.0),
    ("tendon_left", "grip_left", 1.0),
    ("tendon_right", "grip_left", -1.0),
    ("finger_left", "grip_left", -1.0),
    ("finger_right", "grip_left", 1.0),
]

PIECE_RGBA = {
    "white": [0.92, 0.92, 0.85, 1.0],
    "black": [0.10, 0.10, 0.10, 1.0],
}

# Standard chess starting position. file/rank indices: a=0..h=7, rank 1=0..8=7.
BACK_RANK = ["rook", "knight", "bishop", "queen", "king", "bishop", "knight", "rook"]
FILES = "abcdefgh"


def starting_layout():
    """Yield (square_name, file_idx, rank_idx, color, piece_type) for all 32 pieces."""
    for file_idx, ptype in enumerate(BACK_RANK):
        yield (f"{FILES[file_idx]}1", file_idx, 0, "white", ptype)
        yield (f"{FILES[file_idx]}8", file_idx, 7, "black", ptype)
    for file_idx in range(8):
        yield (f"{FILES[file_idx]}2", file_idx, 1, "white", "pawn")
        yield (f"{FILES[file_idx]}7", file_idx, 6, "black", "pawn")


def add_piece(worldbody, board_pos, square_name, file_idx, rank_idx, color, ptype):
    """Add one free-floating piece body, resting on the board surface.

    MuJoCo requires free joints on top-level bodies, so pieces are attached
    directly to worldbody (not nested under the static "chessboard" body) --
    `board_pos` supplies the world offset that the board-local site grid uses.

    Each piece body's origin is its own *bottom* (where it touches the board),
    so every geom below is stacked upward from z=0 in the body's local frame --
    callers that move a piece can just set the body position to a target
    surface point without needing per-piece-type height offsets.
    """
    world = board_pos + np.array([(rank_idx - 3.5) * SQUARE, (file_idx - 3.5) * SQUARE, BOARD_SURFACE_Z])
    body = worldbody.add_body(name=f"piece_{square_name}", pos=world.tolist())
    body.add_freejoint()
    rgba = PIECE_RGBA[color]
    s = SQUARE

    if ptype == "pawn":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.35 * s, 0.45 * s, 0], pos=[0, 0, 0.45 * s], rgba=rgba)
    elif ptype == "rook":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.34 * s, 0.34 * s, 0.5 * s], pos=[0, 0, 0.5 * s], rgba=rgba)
    elif ptype == "knight":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.30 * s, 0.30 * s, 0.45 * s], pos=[0, 0, 0.45 * s], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.16 * s, 0.12 * s, 0.12 * s], pos=[0.16 * s, 0, 0.78 * s], rgba=rgba)
    elif ptype == "bishop":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.30 * s, 0.5 * s, 0], pos=[0, 0, 0.5 * s], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.22 * s, 0, 0], pos=[0, 0, 1.05 * s], rgba=rgba)
    elif ptype == "queen":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.34 * s, 0.55 * s, 0], pos=[0, 0, 0.55 * s], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.30 * s, 0, 0], pos=[0, 0, 1.15 * s], rgba=rgba)
    elif ptype == "king":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.34 * s, 0.60 * s, 0], pos=[0, 0, 0.60 * s], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.26 * s, 0, 0], pos=[0, 0, 1.24 * s], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05 * s, 0.05 * s, 0.18 * s], pos=[0, 0, 1.55 * s], rgba=rgba)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.14 * s, 0.05 * s, 0.05 * s], pos=[0, 0, 1.55 * s], rgba=rgba)
    else:
        raise ValueError(f"unknown piece type {ptype!r}")


def build():
    parent = mujoco.MjSpec.from_file("table_scene.xml")
    child = mujoco.MjSpec.from_file("learm.urdf")

    mount = parent.worldbody.add_frame(pos=ARM_MOUNT_POS)
    parent.attach(child, prefix=ARM_PREFIX, frame=mount)

    # These joints are fully slaved to grip_left via the equality constraints
    # below. Their URDF <limit> ranges were authored for MoveIt planning and
    # conflict with the slaved target, so widen them to avoid the equality
    # solver fighting a redundant limit.
    for dep, _, _ in MIMIC:
        parent.joint(ARM_PREFIX + dep).range = [-3.14, 3.14]

    for dep, master, mult in MIMIC:
        parent.add_equality(
            type=mujoco.mjtEq.mjEQ_JOINT,
            name1=ARM_PREFIX + dep,
            name2=ARM_PREFIX + master,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            data=[0, mult, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

    for jname in CONTROLLED_JOINTS:
        full = ARM_PREFIX + jname
        jnt = parent.joint(full)
        lo, hi = jnt.range
        parent.add_actuator(
            name="act_" + jname,
            target=full,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrlrange=[lo, hi],
            gainprm=[40, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            biasprm=[0, -40, -2, 0, 0, 0, 0, 0, 0, 0],
        )

    hand_body = parent.body(ARM_PREFIX + "hand_link")
    hand_body.add_site(name="gripper_tip", pos=[0, 0, 0.085], size=[0.004, 0.004, 0.004])

    board_body = parent.body("chessboard")
    files = "abcdefgh"
    for fi in range(8):
        for ri in range(8):
            local = np.array([(ri - 3.5) * SQUARE, (fi - 3.5) * SQUARE, BOARD_SURFACE_Z])
            board_body.add_site(
                name=f"sq_{files[fi]}{ri+1}",
                pos=local.tolist(),
                size=[0.002, 0.002, 0.002],
                rgba=[1, 0, 0, 0.6],
            )

    board_pos = np.array(board_body.pos)
    for square_name, file_idx, rank_idx, color, ptype in starting_layout():
        add_piece(parent.worldbody, board_pos, square_name, file_idx, rank_idx, color, ptype)

    model = parent.compile()
    return parent, model


if __name__ == "__main__":
    spec, model = build()
    print("compiled OK: nbody=", model.nbody, "njnt=", model.njnt, "neq=", model.neq, "nu=", model.nu)
    xml = spec.to_xml()
    with open("learm_chess_scene.xml", "w") as f:
        f.write(xml)
    print("wrote learm_chess_scene.xml,", len(xml), "bytes")
