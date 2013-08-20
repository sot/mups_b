from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

torque = np.array([[-1.3911, 1.3977, -1.3766, 1.3699],
                   [-1.8115, -1.8115, 1.8115, 1.8115],
                   [-1.0458, 1.0458, 1.0458, -1.0458]])


def plot_corr_envelope(start_mom=60., target_mom=40., n_grid=2000):
    """
    Plot envelope where it is possible to correct from the given
    start momentum to the given target momentum.
    """
    A_grid = sphere_grid(n_grid)
    t0 = torque[:, 0]
    t1 = torque[:, 2]
    t2 = torque[:, 3]
    t0 *= 1000 / norm(t0)
    t1 *= 1000 / norm(t1)
    t2 *= 1000 / norm(t2)
    opp = -(t0 + t1 + t2)
    opp /= norm(opp)

    planes = [(t0, t1), (t0, t2), (t1, t2)]
    goods = []
    bads = []
    for A_g in A_grid:
        for v0, v1 in planes:
            A = A_g * start_mom
            B = A + v0
            C = A + v1
            if intersect_triangle_sphere(A, B, C, target_mom):
                goods.append(A.tolist())
                break
        else:
            # Hack because of the intersect_triangle algorithm
            # doesn't cover the case of the sphere completely
            # enclosed in the pyramid of influence.
            if np.dot(A_g, opp) > 0.8:
                goods.append(A.tolist())
            else:
                bads.append(A.tolist())
    plt.clf()
    goods = np.array(goods)
    bads = np.array(bads)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(goods[:, 0], goods[:, 1], '.b', zs=goods[:, 2])
    ax.plot(bads[:, 0], bads[:, 1], '.r', zs=bads[:, 2], ms=10)
    plt.draw()
    plt.show()
    print('About {:.1f}% of momentum space is OK (can correct from {} to {})'
          .format(len(goods) / len(A_grid) * 100, start_mom, target_mom))


def norm(A):
    return np.sqrt(np.sum(A ** 2))


def intersect_triangle_sphere(A, B, C, r, P=None):
    if P is None:
        P = [0, 0, 0]
    A = np.array(A, dtype=np.float)
    B = np.array(B, dtype=np.float)
    C = np.array(C, dtype=np.float)
    P = np.array(P, dtype=np.float)

    A = A - P
    B = B - P
    C = C - P
    rr = r * r

    V = np.cross(B - A, C - A)
    d = np.dot(A, V)
    e = np.dot(V, V)
    sep1 = d * d > rr * e
    aa = np.dot(A, A)
    ab = np.dot(A, B)
    ac = np.dot(A, C)
    bb = np.dot(B, B)
    bc = np.dot(B, C)
    cc = np.dot(C, C)
    sep2 = (aa > rr) & (ab > aa) & (ac > aa)
    sep3 = (bb > rr) & (ab > bb) & (bc > bb)
    sep4 = (cc > rr) & (ac > cc) & (bc > cc)
    AB = B - A
    BC = C - B
    CA = A - C
    d1 = ab - aa
    d2 = bc - bb
    d3 = ac - cc
    e1 = np.dot(AB, AB)
    e2 = np.dot(BC, BC)
    e3 = np.dot(CA, CA)
    Q1 = A * e1 - d1 * AB
    Q2 = B * e2 - d2 * BC
    Q3 = C * e3 - d3 * CA
    QC = C * e1 - Q1
    QA = A * e2 - Q2
    QB = B * e3 - Q3
    sep5 = (np.dot(Q1, Q1) > rr * e1 * e1) & (np.dot(Q1, QC) > 0)
    sep6 = (np.dot(Q2, Q2) > rr * e2 * e2) & (np.dot(Q2, QA) > 0)
    sep7 = (np.dot(Q3, Q3) > rr * e3 * e3) & (np.dot(Q3, QB) > 0)
    separated = sep1 | sep2 | sep3 | sep4 | sep5 | sep6 | sep7

    return not separated


def test():
    A = [0, 0, 2.]
    B = [10, 0, 2.]
    C = [0, 10, 2.]
    print intersect_triangle_sphere(A, B, C, 1.5)


def sphere_grid(n_grid):
    """Calculate approximately uniform spherical grid of rays containing
    ``n_grid`` points and extending over the opening angle ``open_angle``
    (radians).

    :returns: np array of unit length rays, grid area (steradians)
    """
    from math import sin, cos, radians, pi, sqrt

    open_angle = pi / 2
    grid_area = 2 * pi
    if n_grid <= 1:
        return np.array([[1., 0., 0.]]), grid_area

    gridsize = sqrt(grid_area / n_grid)

    grid = []
    n_d = int(round(open_angle / gridsize))
    d_d = open_angle / n_d

    for i_d in range(0, n_d + 1):
        dec = i_d * d_d
        if abs(i_d) != n_d:
            n_r = int(round(2 * pi * cos(dec) / d_d))
            d_r = 2 * pi / n_r
        else:
            n_r = 1
            d_r = 1
        for i_r in range(0, n_r):
            ra = i_r * d_r
            # This has x <=> z (switched) from normal formulation to make the
            # grid centered about the x-axis
            grid.append((sin(dec), sin(ra) * cos(dec), cos(ra) * cos(dec)))

    grid = np.array(grid)
    ok = grid[:, 0] > 1e-8
    negx_grid = grid[ok]
    negx_grid[:, 0] = -negx_grid[:, 0]
    return np.vstack((grid, negx_grid))
