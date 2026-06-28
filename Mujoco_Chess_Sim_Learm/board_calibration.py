"""The 2-corner approximation theory: measure a1 and h8, interpolate the rest.

THE IDEA
--------
A chess board is a fixed, uniform grid: every square is the same size and
they're laid out in straight rows/columns. "Fixed spacing" means the distance
between adjacent square centres is constant across the whole board -- it's
guaranteed by how the board is manufactured, not something you need to verify
square by square.

Because of that, two measurements fully determine all 64 positions:
  - a1 tells you the grid's origin.
  - h8 (the diagonal-opposite corner) tells you, combined with a1, both the
    overall span and which direction is "up the board".

If the board is placed with its edges parallel to the robot's x/y axes (the
setup recommended in this project's README -- just align the board edge when
you put it down, it's a one-second physical step), file and rank vary
independently:
    x(rank) = a1.x + (h8.x - a1.x) * rank_idx / 7
    y(file) = a1.y + (h8.y - a1.y) * file_idx / 7
Dividing by 7 (not 8) because a1->h8 spans 7 *gaps* between 8 squares.

This needs the board to be axis-aligned. If it's rotated about z but still a
perfect 8x8 square, square_xyz_rotated() below handles that general case by
exploiting the fact that the a1->h8 diagonal bisects the right angle between
the file-axis and rank-axis -- but it's more sensitive to measurement noise
in a1/h8 than the axis-aligned version, so axis-aligned placement is still
the recommended setup.

Either way: you measure 2 points, not 64.
"""

import numpy as np


def square_xyz(file_idx, rank_idx, a1, h8):
    """Axis-aligned interpolation. file_idx: a=0..h=7, rank_idx: 1=0..8=7.

    Assumes the board's file-direction and rank-direction are parallel to
    the robot's x and y axes respectively (x varies only with rank/file_idx
    correspondence below matches this project's table_scene.xml layout,
    where rank runs along x and file runs along y -- swap if yours differs).
    """
    a1 = np.asarray(a1, dtype=float)
    h8 = np.asarray(h8, dtype=float)
    u = rank_idx / 7.0
    v = file_idx / 7.0
    x = a1[0] + (h8[0] - a1[0]) * u
    y = a1[1] + (h8[1] - a1[1]) * v
    z = (a1[2] + h8[2]) / 2.0
    return np.array([x, y, z])


def square_xyz_rotated(file_idx, rank_idx, a1, h8, normal=(0, 0, 1)):
    """General case: board is a perfect 8x8 square but rotated about `normal`.

    a1->h8 is the board's main diagonal. For a square grid, that diagonal
    bisects the right angle between the file-axis and rank-axis, so rotating
    it by +-45 degrees about the board normal recovers both axes from just
    the 2 corner measurements -- no third point needed, at the cost of being
    more sensitive to measurement error than the axis-aligned formula.
    """
    a1 = np.asarray(a1, dtype=float)
    h8 = np.asarray(h8, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    diag = h8 - a1

    def rotate(v, axis, angle):
        # Rodrigues' rotation formula
        return (v * np.cos(angle)
                + np.cross(axis, v) * np.sin(angle)
                + axis * np.dot(axis, v) * (1 - np.cos(angle)))

    half = 1.0 / np.sqrt(2.0)
    rank_axis_full = rotate(diag, n, -np.pi / 4) * half
    file_axis_full = rotate(diag, n, +np.pi / 4) * half
    pos = a1 + (rank_idx / 7.0) * rank_axis_full + (file_idx / 7.0) * file_axis_full
    return pos


FILES = "abcdefgh"


def square_name(file_idx, rank_idx):
    return f"{FILES[file_idx]}{rank_idx + 1}"


def all_squares(a1, h8, rotated=False, **kwargs):
    """Return {'a1': xyz, 'a2': xyz, ..., 'h8': xyz} for all 64 squares."""
    fn = square_xyz_rotated if rotated else square_xyz
    out = {}
    for file_idx in range(8):
        for rank_idx in range(8):
            out[square_name(file_idx, rank_idx)] = fn(file_idx, rank_idx, a1, h8, **kwargs)
    return out


def validate_against_sim(model_path="learm_chess_scene.xml"):
    """Pretend we only measured a1 and h8 (read from the sim's ground-truth
    sites), interpolate the other 62, and compare against the sim's actual
    site positions -- this is the 'training' check: does the fixed-spacing
    approximation actually hold for this board?
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def truth(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"sq_{name}")
        return data.site_xpos[sid].copy()

    a1, h8 = truth("a1"), truth("h8")
    predicted = all_squares(a1, h8)

    errors_mm = []
    print(f"{'square':8s} {'predicted (mm)':28s} {'truth (mm)':28s} {'error (mm)':>10s}")
    for name, pred in predicted.items():
        actual = truth(name)
        err = np.linalg.norm(pred - actual) * 1000
        errors_mm.append(err)
        if name in ("a1", "h1", "a8", "h8", "d4", "e5"):
            print(f"{name:8s} {np.round(pred*1000,2)!s:28s} {np.round(actual*1000,2)!s:28s} {err:10.3f}")

    errors_mm = np.array(errors_mm)
    print(f"\nmax error: {errors_mm.max():.3f} mm   mean error: {errors_mm.mean():.3f} mm")
    print("(near-zero confirms the fixed-spacing assumption holds for this board;")
    print(" on the real arm, expect mm-scale error from how precisely you jog to a1/h8.)")


if __name__ == "__main__":
    validate_against_sim()
