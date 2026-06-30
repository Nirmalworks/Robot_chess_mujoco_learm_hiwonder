# MuJoCo Chess Sim — LeArm

A MuJoCo simulation of the Hiwonder LeArm (URDF from
[andrewda/learm_ros2](https://github.com/andrewda/learm_ros2), BSD-3-Clause)
mounted at a small chess board on a table, for testing the 2-corner board
calibration method before trying it on the real arm. No camera is mounted
over the board — square positions come from the interpolation method in
[`board_calibration.py`](board_calibration.py), not vision.

A full set of 32 chess pieces is modeled on the board, and
[`view_sim.py`](view_sim.py) has a `--play` mode that lets you play a game
against Stockfish from the terminal — type moves in algebraic notation, your
own piece is just registered (you moved it by hand), and Stockfish's reply
is carried out by the arm's own joints via numerical inverse kinematics.

## Contents

- `learm.urdf`, `meshes/` — LeArm description, copied from upstream with
  `<inertial>` blocks added and a MuJoCo compiler hint (see the comment at
  the top of `learm.urdf` for exactly what changed and why).
- `table_scene.xml` — table + chessboard MJCF, authored directly (no robot).
- `build_scene.py` — attaches `learm.urdf` to `table_scene.xml` via MuJoCo's
  model-composition API (`MjSpec.attach`), adds position actuators, the
  gripper-mimic equality constraints, a `gripper_tip` site, and 64
  ground-truth `sq_a1`..`sq_h8` sites. Writes `learm_chess_scene.xml`.
- `learm_chess_scene.xml` — the generated, self-contained scene. This is
  what actually gets loaded; re-run `build_scene.py` to regenerate it if you
  edit `learm.urdf` or `table_scene.xml`.
- `view_sim.py` — opens the interactive MuJoCo viewer on the scene (manual
  jogging mode), or with `--play`, an interactive chess game against
  Stockfish that's replayed move-by-move in the 3-D scene.
- `board_calibration.py` — the 2-corner interpolation method + a validator
  that checks it against the sim's ground-truth square positions.

## Setup

```bash
pip install -r requirements.txt
```

`--play` mode also needs a `stockfish` binary on `PATH`:

```bash
brew install stockfish        # macOS
# or: apt install stockfish   # Debian/Ubuntu
```

(or pass `--stockfish /path/to/stockfish` to point at one explicitly).

## Run it

```bash
python3 view_sim.py
```

This opens MuJoCo's interactive viewer. Open the **Control** tab (right-hand
panel) to get one slider per joint: `shoulder_pan`, `shoulder_lift`, `elbow`,
`wrist_flex`, `wrist_roll`, `grip_left`. Drag a slider to jog that joint —
the other 5 gripper-linkage joints (`grip_right`, `tendon_left/right`,
`finger_left/right`) mirror `grip_left` automatically, matching how the real
gripper's single servo drives both fingers through a linkage.

Double-click a body (e.g. the hand) to select it and see its live position
in the bottom-left overlay — that's how you read off a1/h8 in sim instead of
on the real arm (see Approximation theory below).

If you change `learm.urdf` (e.g. pull a newer version of the upstream repo)
or `table_scene.xml`, regenerate the combined scene:

```bash
python3 build_scene.py
```

### Playing chess against Stockfish

```bash
python3 view_sim.py --play                       # you play White
python3 view_sim.py --play --side black          # you play Black
python3 view_sim.py --play --skill 5 --movetime 0.5   # weaker, faster engine
```

**On macOS, use `mjpython` instead of `python3` for `--play`**
(`mjpython view_sim.py --play`, same args otherwise) — MuJoCo's passive
viewer, which `--play` needs to run the terminal loop alongside the live
render, requires the main thread on macOS. `mjpython` ships alongside the
`mujoco` pip package. Plain `python3 view_sim.py` (no `--play`, the manual
jogging mode) doesn't need it, since `mujoco.viewer.launch` is a blocking
call with no such restriction.

Type moves in standard algebraic notation at the terminal prompt — `e4`,
`Nf3`, `exd5`, `O-O` — optionally followed by a color word as a sanity check
(`e4 white`); it's compared against whose turn it actually is and the move
is rejected if it doesn't match.

Your move is just registered (matching the real setup: you physically moved
your own piece by hand). Stockfish then replies as "the robot," and **its
move is carried out by the arm itself**, not just teleported: `view_sim.py`'s
`ArmController` runs numerical inverse kinematics (damped-least-squares on
the `gripper_tip` site's Jacobian, position-only — see the class docstring
for why orientation isn't constrained) to drive `shoulder_pan` /
`shoulder_lift` / `elbow` / `wrist_flex` / `wrist_roll` through a hover →
descend → grasp → lift → carry → place → release → retreat sequence for
each piece that needs to move (captures mean the captured piece is picked up
and carried to a holding area beside the board *before* the capturing piece
is placed — two full pick-and-place sequences; castling is also two, king
then rook). The arm's existing position actuators (PD control, see
`build_scene.py`) supply the "slowly move" motion — `ctrl` is ramped toward
each IK solution in small steps rather than jumped, both because that's how
the real Arduino sketch eases servos (1°/15ms, see root README) and because
jumping it outright reliably destabilized the sim during testing (the link
masses are only approximate — see Known limitations — so the PD gains are
stiff relative to them).

There's no real grasp physics yet: the arm has no collision geometry (see
Known limitations), so a piece is kinematically pinned to the `gripper_tip`
site's position (with a small offset) while "held," not actually gripped by
friction — opening/closing the gripper is cosmetic. Giving the arm collision
geometry and a real grasp is the natural next step if this turns out to
matter; RL training is the documented fallback if IK alone proves too
fiddly for the small gripper, per this project's own framing of it as a
step-by-step build.

## Does jogging in the viewer move the real arm?

**No.** This is a pure software simulation — `view_sim.py` only steps
MuJoCo's physics and never opens a serial port. It is completely
disconnected from the Arduino/`Python_Control_Learm` code in this repo.
Dragging a slider here has no effect on the physical servos, and moving the
real arm has no effect on the simulation.

If you want the two connected (sim mirrors hardware, or hardware mirrors
sim), that needs a separate bridge script that reads joint angles each frame
and writes/reads the same serial protocol the Arduino sketch expects
(`base,shoulder,arm,wrist,elbow,gripper\n`, see the root `README.md`). That
bridge doesn't exist yet — say if you want it built.

Also note: this URDF's zero pose (all joints at 0) is the arm fully
extended *upward*, not the real arm's `HOME = [100, 85, 84, 178, 72, 143]`
servo-degree pose from `xbox_controller_arm_control.py`. The two use
different zero-angle conventions, so joint angles don't translate 1:1
between sim and the real servos without a calibration offset per joint —
relevant if you ever do build that bridge.

## Approximation theory: 2-corner board calibration

A chess board is a fixed, uniform grid — every square is the same size,
laid out in straight rows and columns by construction. That regularity is
the whole trick: instead of measuring all 64 square centres, measure only
the two diagonally-opposite corners, **a1** and **h8**, and let linear
interpolation fill in the other 62.

```
x(rank) = a1.x + (h8.x - a1.x) * rank_idx / 7
y(file) = a1.y + (h8.y - a1.y) * file_idx / 7
```

(divide by 7, not 8 — a1→h8 spans 7 gaps between 8 squares). This assumes
the board is placed with its edges parallel to the robot's x/y axes, which
just means aligning the board edge with the table edge when you set it down
— a one-second physical step, and the setup this project's layout assumes.

In sim, "measuring a1 and h8" means jogging the arm until `gripper_tip` (or
the hand body) sits over each square's centre and reading its position from
the viewer overlay — exactly the same workflow you'd use on the real arm,
except the simulated forward kinematics give you exact numbers instead of a
tape measure. Since the URDF used here matches the real LeArm's geometry,
those same two corner readings are valid for the real arm too (once
joint-angle conventions are calibrated — see above).

Run the validator to see the method checked against the sim's own
ground-truth square positions:

```bash
python3 board_calibration.py
```

It "measures" a1/h8 from the sim's `sq_a1`/`sq_h8` sites, interpolates all
64 squares, and compares against the sim's actual site positions. Expect
~0 mm error here (the simulated board *is* a perfect uniform grid by
construction) — the real-world error will instead come from how precisely
you jog onto a1/h8 by eye.

`board_calibration.py` also has `square_xyz_rotated()`, a more general
version for when the board can't be perfectly axis-aligned (uses the fact
that the a1→h8 diagonal bisects the angle between the file- and rank-axes
on a square grid) — more sensitive to measurement noise, so axis-aligned
placement is still the recommended setup.

## Known limitations

- **Approximate inertials**: the upstream URDF has no mass/inertia data.
  Values here are derived from each mesh's volume × an assumed effective
  density (see the comment in `learm.urdf`) — good enough for kinematic
  positioning, not for accurate torque/force dynamics.
- **No collision geometry on the arm, so no real grasp physics**: upstream
  has no `<collision>` elements either, so the arm can't physically contact
  the chess pieces in sim yet. During `--play`, the robot's (Stockfish's)
  moves *are* executed by the arm's own joints via inverse kinematics (see
  `ArmController` in `view_sim.py`) — it's not a position-only replay
  anymore — but a piece is kinematically pinned to the `gripper_tip` site
  while "held" rather than actually gripped by friction/contact. Giving the
  arm collision geometry for a real grasp is the natural next step; if that
  (or the IK itself) proves too fiddly for this small gripper, training
  (RL) is the documented fallback, per the project's own framing of this as
  a step-by-step build.
- **IK is position-only, no orientation control**: the 5 arm joints feed a
  3-DOF position task (already redundant), solved with damped-least-squares
  on the `gripper_tip` Jacobian. The gripper's *orientation* at each
  waypoint is whatever falls out of that solve, not constrained to point
  straight down — fine for a cosmetic pick-and-place, but worth knowing if
  you extend this toward a real grasp controller.
- **Board/mount placement is tuned to this arm's reach, not a free choice**:
  `ARM_MOUNT_POS` in `build_scene.py` and the chessboard position/size (in
  `build_scene.py` / `table_scene.xml`) are set so the near edge of the
  board sits exactly 0.15 m from the arm's mount point — matching the real
  hardware's robot-to-board gap. The board's square size (0.018 m, vs. a
  more standard ~0.025 m) was shrunk from an earlier, closer-mounted version
  specifically so the far rank (8) still stays within this small arm's
  reach at that 0.15 m gap (its max horizontal reach is only ~0.27-0.28 m at
  board height) — verified by sampling reachability for all 4 corners. If
  your real table layout differs, re-verify reachability before trusting
  these constants; a bigger board pushed out to a true 0.15 m gap will not
  fit this arm's reach.
