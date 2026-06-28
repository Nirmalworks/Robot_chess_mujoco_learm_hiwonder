import mujoco
import numpy as np

ARM_PREFIX = "arm_"
ARM_MOUNT_POS = (-0.03, 0.0, 0.0)   # base_link position on the tabletop
SQUARE = 0.025                       # must match the board size in table_scene.xml (0.20 m / 8)

CONTROLLED_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_flex", "wrist_roll", "grip_left"]
MIMIC = [
    # (dependent_joint, master_joint, multiplier)
    ("grip_right", "grip_left", -1.0),
    ("tendon_left", "grip_left", 1.0),
    ("tendon_right", "grip_left", -1.0),
    ("finger_left", "grip_left", -1.0),
    ("finger_right", "grip_left", 1.0),
]


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
            local = np.array([(ri - 3.5) * SQUARE, (fi - 3.5) * SQUARE, 0.0046])
            board_body.add_site(
                name=f"sq_{files[fi]}{ri+1}",
                pos=local.tolist(),
                size=[0.002, 0.002, 0.002],
                rgba=[1, 0, 0, 0.6],
            )

    model = parent.compile()
    return parent, model


if __name__ == "__main__":
    spec, model = build()
    print("compiled OK: nbody=", model.nbody, "njnt=", model.njnt, "neq=", model.neq, "nu=", model.nu)
    xml = spec.to_xml()
    with open("learm_chess_scene.xml", "w") as f:
        f.write(xml)
    print("wrote learm_chess_scene.xml,", len(xml), "bytes")
