"""Launch the interactive MuJoCo viewer on the LeArm + chess table scene.

Run:
    python3 view_sim.py

In the viewer window, open the right-hand "Control" tab to find one slider
per joint (shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll,
grip_left) -- drag a slider to jog that joint. The gripper fingers mirror
grip_left automatically. This only drives the simulation; see README.md for
why it does NOT move the real hardware.

Hold Ctrl and drag with the mouse to apply a force/torque to a body, or
double-click a body to select it (its name + live xyz/quat show in the
bottom-left overlay) -- this is the easiest way to read off a position when
you jog the arm onto square a1 or h8 for the corner-calibration method in
board_calibration.py.
"""

import mujoco
import mujoco.viewer

MODEL_PATH = "learm_chess_scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
