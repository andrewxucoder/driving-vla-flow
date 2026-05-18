import numpy as np

from driving_vla.data.coordinate_transform import ego_to_global_frame, global_to_ego_frame


def test_origin_pose_does_not_change_points():
    points = np.array([[1.0, 2.0], [3.0, -1.0]])
    ego_pose = np.array([0.0, 0.0, 0.0])

    transformed = global_to_ego_frame(points, ego_pose)

    np.testing.assert_allclose(transformed, points)


def test_translation_centers_points_around_ego():
    points = np.array([[10.0, 5.0], [12.0, 8.0]])
    ego_pose = np.array([10.0, 5.0, 0.0])

    transformed = global_to_ego_frame(points, ego_pose)

    np.testing.assert_allclose(transformed, np.array([[0.0, 0.0], [2.0, 3.0]]))


def test_rotation_aligns_ego_heading_with_x_axis():
    points = np.array([[0.0, 1.0]])
    ego_pose = np.array([0.0, 0.0, np.pi / 2])

    transformed = global_to_ego_frame(points, ego_pose)

    np.testing.assert_allclose(transformed, np.array([[1.0, 0.0]]), atol=1e-8)


def test_batched_transform_round_trip():
    points = np.array([
        [[1.0, 0.0], [2.0, 0.0]],
        [[0.0, 2.0], [0.0, 3.0]],
    ])
    ego_pose = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, np.pi / 2],
    ])

    ego_points = global_to_ego_frame(points, ego_pose)
    restored = ego_to_global_frame(ego_points, ego_pose)

    np.testing.assert_allclose(restored, points, atol=1e-8)
