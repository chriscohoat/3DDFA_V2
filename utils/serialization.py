# coding: utf-8

__author__ = 'cleardusk'

import numpy as np

from utils.pose import calc_pose

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


def create_sphere(cx, cy, cz, r, resolution=360, phi_resolution=360, theta_resolution=360):
    '''
    create sphere with center (cx, cy, cz) and radius r
    '''
    phi = np.linspace(0, 2*np.pi, 2*(phi_resolution if phi_resolution else resolution))
    theta = np.linspace(0, np.pi, theta_resolution if theta_resolution else resolution)

    theta, phi = np.meshgrid(theta, phi)

    r_xy = r*np.sin(theta)
    x = cx + np.cos(phi) * r_xy
    y = cy + np.sin(phi) * r_xy
    z = cz + r * np.cos(theta)

    return x, y, z


def rotate_point_in_3d(point, origin, pitch, yaw, roll):
    '''
    Given a point, rotates the point around an origin in 3D
    '''
    x, y, z = point
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    roll = np.deg2rad(roll)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                   [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    o = np.atleast_2d(origin)
    p = np.atleast_2d(point)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def create_circle(cx, cy, cz, rotate_around_point, pitch, yaw, roll, r, resolution=360, view='front'):
    '''
    create circle with center (cx, cy, cz) and radius r
    '''
    theta = np.linspace(0, 2*np.pi, resolution)
    if not view or view == 'front':
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        z = cz
    elif view == 'side':
        x = cx
        y = cy + r * np.cos(theta)
        z = cz + r * np.sin(theta)
    return rotate_point_in_3d((x, y, z), rotate_around_point, pitch, yaw, roll)


def rotate_line(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def plot_simple_line(ax, start_point, end_point, color="black"):
    '''
    Plot a simple line between two points
    '''
    ax.plot(
        (
            start_point[0],
            end_point[0]
        ),
        (
            start_point[1],
            end_point[1]
        ),
        (
            start_point[2],
            end_point[2]
        ),
        color=color
    )

def get_line_center_point(line):
    '''
    Get the center point of a line.
    line = [start_point, end_point]
    Returns a point in X, Y, Z space.
    '''
    return (
        (line[0][0] + line[1][0]) / 2,
        (line[0][1] + line[1][1]) / 2,
        (line[0][2] + line[1][2]) / 2,
    )

def ser_to_simple_obj_multiple(img, param_lst, ver_lst, tri, height, wfp):
    n_obj = len(ver_lst)  # count obj

    if n_obj <= 0:
        return

    n_vertex = ver_lst[0].shape[1]
    n_face = tri.shape[0]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_proj_type('ortho')

    ax.set_xlim(300, 800)
    ax.set_ylim(100, 600)
    ax.set_zlim(-200, 300)

    with open(wfp, 'w') as f:

        zipped_pose = zip(param_lst, ver_lst)

        for i in range(n_obj):
            param, ver = list(zipped_pose)[i]
            P, pose = calc_pose(param)
            face_pitch = pose[1]
            face_yaw = pose[0]
            face_roll = pose[2]
            
            ver = ver_lst[i]
            
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

            def get_points_for_grouping(index, as_points=False):
                '''
                This function returns the points for a given grouping.
                Normally, it returns the points as an array of arrays.
                If as_points is set to True, it returns the points as a list of points.

                Example return result:

                nose points:
                    [546.5509  546.80835 547.1167  547.1211 ] (x)
                    [361.8166  382.8972  404.28387 422.21112] (y)
                    [155.2895  169.74472 184.611   187.74591] (z)

                If as_points is True:

                nose points:
                    (546.5509, 361.8166, 155.2895)
                    (546.80835, 382.8972, 169.74472)
                    (547.1167, 404.28387, 184.611)
                    (547.1211, 422.21112, 187.74591)
                '''
                l, r = nums[index], nums[index + 1]
                x = ver_lst[i][0, l:r]
                y = ver_lst[i][1, l:r]
                z = ver_lst[i][2, l:r]
                if as_points:
                    vertices = zip(x, y, z)
                    return list(vertices)
                return (
                    x, y, z
                )

            for ind in range(len(nums) - 1):
                f.write('# {}\n'.format(corresponding_groupings[ind]))
                l, r = nums[ind], nums[ind + 1]
                sphere_x = ver_lst[i][0, l:r]
                sphere_y = ver_lst[i][1, l:r]
                sphere_z = ver_lst[i][2, l:r]
                vertices = zip(sphere_x, sphere_y, sphere_z)
                for (sphere_x, sphere_y, sphere_z) in vertices:
                    f.write(
                        f'v {sphere_x:.2f} {height - sphere_y:.2f} {sphere_z:.2f}\n')
                    if ind == 1:
                        print(
                            f'v {sphere_x:.2f} {sphere_y:.2f} {sphere_z:.2f}\n')
                f.write('\n')
                line_groupings.append((sphere_x, sphere_y, sphere_z))
                ax.plot(sphere_x, sphere_y, sphere_z,
                        label=corresponding_groupings[ind])

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
            # for s, e in combinations(np.array(list(product(r, r, r))), 2):
            #    if np.sum(np.abs(s-e)) == r[1]-r[0]:
            #        ax.plot3D(*zip(s, e), color="b")

            # draw sphere

            # get the points that correspond to the nose in the Z direction

            nose_points = list(get_points_for_grouping(
                3)[2]) + list(get_points_for_grouping(4)[2])

            # find the largest point in the Z direction

            max_nose_z = max(nose_points)

            f.write('# {}\n'.format(corresponding_groupings[ind]))
            right_eyebrow_points = get_points_for_grouping(1)
            left_eyebrow_points = get_points_for_grouping(2)
            eyebrow_centroid = (
                sum(right_eyebrow_points[0] + left_eyebrow_points[0]) /
                (len(right_eyebrow_points[0]) + len(left_eyebrow_points[0])),
                sum(right_eyebrow_points[1] + left_eyebrow_points[1]) /
                (len(right_eyebrow_points[1]) + len(left_eyebrow_points[1])),
                sum(right_eyebrow_points[2] + left_eyebrow_points[2]) /
                (len(right_eyebrow_points[2]) + len(left_eyebrow_points[2])),
            )

            outer_lips_points = get_points_for_grouping(8)
            lips_centroid = (
                sum(outer_lips_points[0]) / (len(outer_lips_points[0])),
                sum(outer_lips_points[1]) / (len(outer_lips_points[1])),
                sum(outer_lips_points[2]) / (len(outer_lips_points[2])),
            )

            jaw_points = get_points_for_grouping(0)

            ax.plot(
                (jaw_points[0][0], jaw_points[0][-1]),
                (jaw_points[1][0], jaw_points[1][-1]),
                (jaw_points[2][0], jaw_points[2][-1]),
                color="black"
            )

            # Find the center of a line between the two points

            jaw_point1 = (jaw_points[0][0], jaw_points[1][0], jaw_points[2][0])
            jaw_point2 = (jaw_points[0][-1],
                          jaw_points[1][-1], jaw_points[2][-1])

            jaw_back_line_center_point = (
                (jaw_point1[0] + jaw_point2[0]) / 2,
                (jaw_point1[1] + jaw_point2[1]) / 2,
                (jaw_point1[2] + jaw_point2[2]) / 2,
            )

            ax.scatter(
                jaw_back_line_center_point[0], jaw_back_line_center_point[1], jaw_back_line_center_point[2], color="red")

            sphere_radius = abs(lips_centroid[1] - eyebrow_centroid[1])
            sphere_x, sphere_y, sphere_z = create_sphere(
                jaw_back_line_center_point[0],
                jaw_back_line_center_point[1],
                max_nose_z - sphere_radius,
                sphere_radius,
                resolution=10
            )

            # The following code finds all of the sphere points that are within the jaw

            # get the points that correspond to the jaw in the X direction
            jaw_points_x = list(jaw_points[0])
            max_jaw_x = max(jaw_points_x)
            min_jaw_x = min(jaw_points_x)
            # get the points that correspond to the jaw in the Y direction
            jaw_points_y = list(get_points_for_grouping(0)[1])
            min_jaw_Y = min(jaw_points_y)

            # unzip sphere points

            sphere_points = list(
                zip(sphere_x.flatten(), sphere_y.flatten(), sphere_z.flatten()))

            # remove the points where X is greater than max_jaw_x and less than min_jaw_x

            sphere_points_inside_jaw = [point for point in sphere_points if point[0]
                                        <= max_jaw_x and point[0] >= min_jaw_x and point[1] <= min_jaw_Y]

            # Zip sphere points to X, Y, and Z

            sphere_x_inside_jaw, sphere_y_inside_jaw, sphere_z_inside_jaw = zip(
                *sphere_points_inside_jaw)

            # ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="red") # plot_surface is a function also
            # ax.scatter(np.array(sphere_x_inside_jaw), np.array(sphere_y_inside_jaw), np.array(sphere_z_inside_jaw), color="black")

            # 2d only:
            # ax.scatter(sphere_x, sphere_y, color="red") # plot_surface is a function also
            # ax.scatter(np.array(sphere_x_inside_jaw), np.array(sphere_y_inside_jaw), color="orange") # plot_surface is a function also

            rectangle_half_height = 2 * (sphere_radius) / 3 # 2/3 of the sphere diameter

            left_cutoff_line_horizontal = [
                (jaw_point1[0], jaw_point1[1], jaw_point1[2] - rectangle_half_height),
                (jaw_point1[0], jaw_point1[1], jaw_point1[2] + rectangle_half_height),
            ]
            left_cutoff_line_vertical = [
                (jaw_point1[0], jaw_point1[1] - rectangle_half_height, jaw_point1[2]),
                (jaw_point1[0], jaw_point1[1] + rectangle_half_height, jaw_point1[2]),
            ]

            left_cutoff_line_line_center_point = get_line_center_point(left_cutoff_line_horizontal)

            right_cutoff_line_horizontal = [
                (jaw_point2[0], jaw_point2[1], jaw_point2[2] - rectangle_half_height),
                (jaw_point2[0], jaw_point2[1], jaw_point2[2] + rectangle_half_height),
            ]
            right_cutoff_line_vertical = [
                (jaw_point2[0], jaw_point2[1] - rectangle_half_height, jaw_point2[2]),
                (jaw_point2[0], jaw_point2[1] + rectangle_half_height, jaw_point2[2]),
            ]

            right_cutoff_line_line_center_point = get_line_center_point(right_cutoff_line_horizontal)

            # plot_simple_line(ax, left_cutoff_line_horizontal[0], left_cutoff_line_horizontal[1], color="black")
            # plot_simple_line(ax, left_cutoff_line_vertical[0], left_cutoff_line_vertical[1], color="black")
            # plot_simple_line(ax, right_cutoff_line_horizontal[0], right_cutoff_line_horizontal[1], color="black")
            # plot_simple_line(ax, right_cutoff_line_vertical[0], right_cutoff_line_vertical[1], color="black")

            # Rotate left_cutoff_line around a point in 3D

            print(face_yaw, '..', face_pitch, '...', face_roll)
            yaw = -face_yaw
            pitch = face_pitch
            roll = face_roll

            rotated_left_cutoff_line_horizontal_start = rotate_point_in_3d(left_cutoff_line_horizontal[0], left_cutoff_line_line_center_point, pitch, yaw, roll)
            rotated_left_cutoff_line_horizontal_end = rotate_point_in_3d(left_cutoff_line_horizontal[1], left_cutoff_line_line_center_point, pitch, yaw, roll)
            rotated_left_cutoff_line_vertical_start = rotate_point_in_3d(left_cutoff_line_vertical[0], left_cutoff_line_line_center_point, pitch, yaw, roll)
            rotated_left_cutoff_line_vertical_end = rotate_point_in_3d(left_cutoff_line_vertical[1], left_cutoff_line_line_center_point, pitch, yaw, roll)

            rotated_right_cutoff_line_horizontal_start = rotate_point_in_3d(right_cutoff_line_horizontal[0], right_cutoff_line_line_center_point, pitch, yaw, roll)
            rotated_right_cutoff_line_horizontal_end = rotate_point_in_3d(right_cutoff_line_horizontal[1], right_cutoff_line_line_center_point, pitch, yaw, roll)
            rotated_right_cutoff_line_vertical_start = rotate_point_in_3d(right_cutoff_line_vertical[0], right_cutoff_line_line_center_point, pitch, yaw, roll)
            rotated_right_cutoff_line_vertical_end = rotate_point_in_3d(right_cutoff_line_vertical[1], right_cutoff_line_line_center_point, pitch, yaw, roll)

            plot_simple_line(ax, rotated_left_cutoff_line_horizontal_start, rotated_left_cutoff_line_horizontal_end, color="blue")
            plot_simple_line(ax, rotated_left_cutoff_line_vertical_start, rotated_left_cutoff_line_vertical_end, color="blue")
            plot_simple_line(ax, rotated_right_cutoff_line_horizontal_start, rotated_right_cutoff_line_horizontal_end, color="blue")
            plot_simple_line(ax, rotated_right_cutoff_line_vertical_start, rotated_right_cutoff_line_vertical_end, color="blue")

            front_circle_x, front_circle_y, front_circle_z = create_circle(
                jaw_back_line_center_point[0],
                jaw_back_line_center_point[1],
                jaw_back_line_center_point[2],
                jaw_back_line_center_point,
                pitch, yaw, roll,
                sphere_radius,
                resolution=360,
                view='front',
            )
            ax.scatter(front_circle_x, front_circle_y, front_circle_z, color="blue")

            right_circle_x, right_circle_y, right_circle_z = create_circle(
                right_cutoff_line_line_center_point[0],
                right_cutoff_line_line_center_point[1],
                right_cutoff_line_line_center_point[2],
                right_cutoff_line_line_center_point,
                pitch, yaw, roll,
                rectangle_half_height,
                resolution=360,
                view='side',
            )
            ax.scatter(right_circle_x, right_circle_y, right_circle_z, color="blue")
            
            left_circle_x, left_circle_y, left_circle_z = create_circle(
                left_cutoff_line_line_center_point[0],
                left_cutoff_line_line_center_point[1],
                left_cutoff_line_line_center_point[2],
                left_cutoff_line_line_center_point,
                pitch, yaw, roll,
                rectangle_half_height,
                resolution=360,
                view='side',
            )
            ax.scatter(left_circle_x, left_circle_y, left_circle_z, color="blue")

            # Get all zipped_left_circle_points that have Z greater than left_cutoff_line_line_center_point

            # zipped_left_circle_points = zip(left_circle_x, left_circle_y, left_circle_z)
            # zipped_left_circle_points = [x for x in zipped_left_circle_points if x[0] > left_cutoff_line_line_center_point[0] and x[2] > left_cutoff_line_line_center_point[2]]
            # zipped_left_circle_points_x, zipped_left_circle_points_y, zipped_left_circle_points_z = zip(*zipped_left_circle_points)
            # ax.scatter(zipped_left_circle_points_x, zipped_left_circle_points_y, zipped_left_circle_points_z, color="yellow")

            
    ax.view_init(90, 90)
    # ax.view_init(90, 180)

    plt.show()


ser_to_ply = ser_to_ply_multiple  # ser_to_ply_single
ser_to_obj = ser_to_obj_multiple  # ser_to_obj_multiple
