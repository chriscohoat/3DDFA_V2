# coding: utf-8

__author__ = 'cleardusk'

import numpy as np

from .tddfa_util import _to_ctype
from .functions import get_suffix
import matplotlib.pyplot as plt
from itertools import product, combinations

header_temp = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
element face {}
property list uchar int vertex_indices
end_header
"""


def ser_to_ply_single(ver_lst, tri, height, wfp, reverse=True):
    suffix = get_suffix(wfp)

    for i, ver in enumerate(ver_lst):
        wfp_new = wfp.replace(suffix, f'_{i + 1}{suffix}')

        n_vertex = ver.shape[1]
        n_face = tri.shape[0]
        header = header_temp.format(n_vertex, n_face)

        with open(wfp_new, 'w') as f:
            f.write(header + '\n')
            for i in range(n_vertex):
                x, y, z = ver[:, i]
                if reverse:
                    f.write(f'{x:.2f} {height-y:.2f} {z:.2f}\n')
                else:
                    f.write(f'{x:.2f} {y:.2f} {z:.2f}\n')
            for i in range(n_face):
                idx1, idx2, idx3 = tri[i]  # m x 3
                if reverse:
                    f.write(f'3 {idx3} {idx2} {idx1}\n')
                else:
                    f.write(f'3 {idx1} {idx2} {idx3}\n')

        print(f'Dump tp {wfp_new}')


def ser_to_ply_multiple(ver_lst, tri, height, wfp, reverse=True):
    n_ply = len(ver_lst)  # count ply

    if n_ply <= 0:
        return

    n_vertex = ver_lst[0].shape[1]
    n_face = tri.shape[0]
    header = header_temp.format(n_vertex * n_ply, n_face * n_ply)

    with open(wfp, 'w') as f:
        f.write(header + '\n')

        for i in range(n_ply):
            ver = ver_lst[i]
            for j in range(n_vertex):
                x, y, z = ver[:, j]
                if reverse:
                    f.write(f'{x:.2f} {height - y:.2f} {z:.2f}\n')
                else:
                    f.write(f'{x:.2f} {y:.2f} {z:.2f}\n')

        for i in range(n_ply):
            offset = i * n_vertex
            for j in range(n_face):
                idx1, idx2, idx3 = tri[j]  # m x 3
                if reverse:
                    f.write(
                        f'3 {idx3 + offset} {idx2 + offset} {idx1 + offset}\n')
                else:
                    f.write(
                        f'3 {idx1 + offset} {idx2 + offset} {idx3 + offset}\n')

    print(f'Dump tp {wfp}')


def get_colors(img, ver):
    h, w, _ = img.shape
    ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
    ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
    ind = np.round(ver).astype(np.int32)
    colors = img[ind[1, :], ind[0, :], :] / 255.  # n x 3

    return colors.copy()


def ser_to_obj_single(img, ver_lst, tri, height, wfp):
    suffix = get_suffix(wfp)

    n_face = tri.shape[0]
    for i, ver in enumerate(ver_lst):
        colors = get_colors(img, ver)

        n_vertex = ver.shape[1]

        wfp_new = wfp.replace(suffix, f'_{i + 1}{suffix}')

        with open(wfp_new, 'w') as f:
            for i in range(n_vertex):
                x, y, z = ver[:, i]
                f.write(
                    f'v {x:.2f} {height - y:.2f} {z:.2f} {colors[i, 2]:.2f} {colors[i, 1]:.2f} {colors[i, 0]:.2f}\n')
            for i in range(n_face):
                idx1, idx2, idx3 = tri[i]  # m x 3
                f.write(f'f {idx3 + 1} {idx2 + 1} {idx1 + 1}\n')

        print(f'Dump tp {wfp_new}')


def ser_to_obj_multiple(img, ver_lst, tri, height, wfp):
    n_obj = len(ver_lst)  # count obj

    if n_obj <= 0:
        return

    n_vertex = ver_lst[0].shape[1]
    n_face = tri.shape[0]

    with open(wfp, 'w') as f:
        for i in range(n_obj):
            ver = ver_lst[i]
            colors = get_colors(img, ver)

            for j in range(n_vertex):
                x, y, z = ver[:, j]
                f.write(
                    f'v {x:.2f} {height - y:.2f} {z:.2f} {colors[j, 2]:.2f} {colors[j, 1]:.2f} {colors[j, 0]:.2f}\n')

        for i in range(n_obj):
            offset = i * n_vertex
            for j in range(n_face):
                idx1, idx2, idx3 = tri[j]  # m x 3
                f.write(
                    f'f {idx3 + 1 + offset} {idx2 + 1 + offset} {idx1 + 1 + offset}\n')

    print(f'Dump tp {wfp}')


def create_sphere(cx, cy, cz, r, resolution=360):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2*np.pi, 2*resolution)
    theta = np.linspace(0, np.pi, resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r*np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return x, y, z


def ser_to_simple_obj_multiple(img, ver_lst, tri, height, wfp):
    n_obj = len(ver_lst)  # count obj

    if n_obj <= 0:
        return

    n_vertex = ver_lst[0].shape[1]
    n_face = tri.shape[0]

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    with open(wfp, 'w') as f:

        for i in range(n_obj):
            ver = ver_lst[i]
            print(len(ver))
            print(n_vertex)

            '''
            '''
            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
            corresponding_groupings = [
                "jaw",
                "right eyebrow",
                "left eyebrow",
                "nose vertical",
                "nose bottom",
                "right eye",
                "left eye",
                "lips boundary",
                "inner lips"
            ]
            line_groupings = []

            def get_points_for_grouping(index):
                l, r = nums[index], nums[index + 1]
                return (
                    ver_lst[i][0, l:r],
                    ver_lst[i][1, l:r],
                    ver_lst[i][2, l:r]
                )

            for ind in range(len(nums) - 1):
                f.write('# {}\n'.format(corresponding_groupings[ind]))
                l, r = nums[ind], nums[ind + 1]
                x = ver_lst[i][0, l:r]
                y = ver_lst[i][1, l:r]
                z = ver_lst[i][2, l:r]
                vertices = zip(x, y, z)
                for (x, y, z) in vertices:
                    f.write(f'v {x:.2f} {height - y:.2f} {z:.2f}\n')
                    if ind == 1:
                        print(f'v {x:.2f} {y:.2f} {z:.2f}\n')
                f.write('\n')
                line_groupings.append((x, y, z))
                ax.plot(x, y, z, label=corresponding_groupings[ind])

                # Connecting lines
                plt.plot(ver_lst[i][0, l:r], ver_lst[i]
                         [1, l:r], ver_lst[i][2, l:r], color='red')
                # Individual points
                plt.plot(ver_lst[i][0, l:r], ver_lst[i][1, l:r],
                         ver_lst[i][2, l:r], marker='o', linestyle='None')

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                print()
                print('ind: {}'.format(ind))
                print(l, '-', r)
                f.write('# {}\n'.format(corresponding_groupings[ind]))
                f.write('l {}\n'.format(
                    " ".join(["{}".format(l + item + 1) for item in range(r - l)])))

            # draw cube
            #r = [-1, 1]
            #for s, e in combinations(np.array(list(product(r, r, r))), 2):
            #    if np.sum(np.abs(s-e)) == r[1]-r[0]:
            #        ax.plot3D(*zip(s, e), color="b")

            # draw sphere

            f.write('# {}\n'.format(corresponding_groupings[ind]))
            right_eyebrow_points = get_points_for_grouping(1)
            left_eyebrow_points = get_points_for_grouping(2)
            eyebrow_centroid = (
                sum(right_eyebrow_points[0] + left_eyebrow_points[0]) / (len(right_eyebrow_points[0]) + len(left_eyebrow_points[0])), 
                sum(right_eyebrow_points[1] + left_eyebrow_points[1]) / (len(right_eyebrow_points[1]) + len(left_eyebrow_points[1])), 
                sum(right_eyebrow_points[2] + left_eyebrow_points[2]) / (len(right_eyebrow_points[2]) + len(left_eyebrow_points[2])), 
            )

            outer_lips_points = get_points_for_grouping(8)
            lips_centroid = (
                sum(outer_lips_points[0]) / (len(outer_lips_points[0])), 
                sum(outer_lips_points[1]) / (len(outer_lips_points[1])), 
                sum(outer_lips_points[2]) / (len(outer_lips_points[2])), 
            )

            print('eyebrow_centroid: ')
            print(eyebrow_centroid)

            x, y, z = create_sphere(eyebrow_centroid[0], eyebrow_centroid[1], eyebrow_centroid[2], abs(lips_centroid[1] - eyebrow_centroid[1]), resolution=8)
            ax.plot_wireframe(x, y, z, color="r")

    ax.view_init(90, 90)

    plt.show()


ser_to_ply = ser_to_ply_multiple  # ser_to_ply_single
ser_to_obj = ser_to_obj_multiple  # ser_to_obj_multiple
ser_to_simple_obj = ser_to_simple_obj_multiple  # ser_to_obj_multiple
