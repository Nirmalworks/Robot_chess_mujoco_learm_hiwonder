# MuJoCo Chess Sim — LeArm

A MuJoCo simulation of the Hiwonder LeArm (URDF from
[andrewda/learm_ros2](https://github.com/andrewda/learm_ros2), BSD-3-Clause)
mounted at a small chess board on a table, for testing the 2-corner board
calibration method before trying it on the real arm. No camera is mounted
over the board — square positions come from the interpolation method in
[`board_calibration.py`](board_calibration.py), not vision.

A full set of 32 chess pieces is modeled on the board, and
[`view_sim.py`](view_sim.py) has a `--play` mode that lets you play a game
against Stockfish from the terminal — type moves in algebraic notation,
Stockfish replies as "the robot," and every move is replayed in the 3-D
scene by moving the corresponding piece body.

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

Type moves in standard algebraic notation at the terminal prompt — `e4`,
`Nf3`, `exd5`, `O-O` — optionally followed by a color word as a sanity check
(`e4 white`); it's compared against whose turn it actually is and the move
is rejected if it doesn't match. After your move, Stockfish (playing "the
robot's" side) replies automatically, and both moves are animated in the
MuJoCo viewer by lifting the corresponding piece body and setting it down on
its destination square (captures go to a small holding area beside the
board; castling moves the rook too).

**This animates the *game state*, not an IK-planned pick-and-place by the
arm's own joints.** The arm has no collision geometry yet (see Known
limitations) and isn't driven during play mode — it just sits parked out of
the way. Making the physical arm actually execute these moves (inverse
kinematics for reach + orientation, grasping, possibly RL training if direct
IK proves too fiddly for the small gripper) is the natural next step, not
attempted here.

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
- **No collision geometry on the arm**: upstream has no `<collision>`
  elements either, so the arm currently can't physically interact with the
  chess pieces in sim. During `--play`, moves are replayed by directly
  setting each piece body's position (a kinematic "replay," not a grasp) —
  the arm itself just sits parked out of the way. Giving the arm collision
  geometry and an IK/grasp controller so it actually executes these moves is
  the natural next step; if direct IK turns out to be too fiddly for this
  small gripper, training (RL) is the fallback, per the project's own
  framing of this as a step-by-step build.
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
