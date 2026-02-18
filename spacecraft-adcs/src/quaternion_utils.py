"""
Quaternion Utilities for Attitude Representation
=================================================
Quaternion convention: q = [q0, q1, q2, q3] where q0 is the scalar part.
This is the Hamilton convention: q = q0 + q1*i + q2*j + q3*k

All functions use numpy arrays of shape (4,) for quaternions
and (3,) for vectors.
"""

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion to unit magnitude."""
    n = np.linalg.norm(q)
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton quaternion product: q1 ⊗ q2.

    q1 = [s1, v1], q2 = [s2, v2]
    q1⊗q2 = [s1*s2 - v1·v2, s1*v2 + s2*v1 + v1×v2]
    """
    s1, v1 = q1[0], q1[1:4]
    s2, v2 = q2[0], q2[1:4]

    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)

    return np.array([s, v[0], v[1], v[2]])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: q* = [q0, -q1, -q2, -q3]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """Quaternion inverse (for unit quaternion, same as conjugate)."""
    return quat_conjugate(q) / np.dot(q, q)


def quat_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """
    Compute the quaternion error: q_err = q_target^(-1) ⊗ q_current.

    The vector part of q_err represents the rotation error axis
    scaled by sin(θ/2).
    """
    q_err = quat_multiply(quat_inverse(q_target), q_current)

    # Convention: ensure scalar part is positive for shortest-path rotation
    if q_err[0] < 0:
        q_err = -q_err

    return q_err


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion to 3x3 rotation matrix (body-to-inertial).

    R = (q0^2 - ||v||^2) I + 2 v v^T + 2 q0 [v]_x
    """
    q = quat_normalize(q)
    q0, q1, q2, q3 = q

    R = np.array([
        [1 - 2*(q2**2 + q3**2),   2*(q1*q2 - q0*q3),     2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),       1 - 2*(q1**2 + q3**2),  2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),       2*(q2*q3 + q0*q1),      1 - 2*(q1**2 + q2**2)]
    ])
    return R


def rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to unit quaternion using Shepperd's method."""
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2, 1] - R[1, 2]) * s
        q2 = (R[0, 2] - R[2, 0]) * s
        q3 = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q0 = (R[2, 1] - R[1, 2]) / s
            q1 = 0.25 * s
            q2 = (R[0, 1] + R[1, 0]) / s
            q3 = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q0 = (R[0, 2] - R[2, 0]) / s
            q1 = (R[0, 1] + R[1, 0]) / s
            q2 = 0.25 * s
            q3 = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q0 = (R[1, 0] - R[0, 1]) / s
            q1 = (R[0, 2] + R[2, 0]) / s
            q2 = (R[1, 2] + R[2, 1]) / s
            q3 = 0.25 * s

    q = np.array([q0, q1, q2, q3])
    return quat_normalize(q)


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Create a quaternion from an axis-angle representation."""
    axis = axis / np.linalg.norm(axis)
    half_angle = angle_rad / 2.0
    q0 = np.cos(half_angle)
    q_vec = axis * np.sin(half_angle)
    return np.array([q0, q_vec[0], q_vec[1], q_vec[2]])


def omega_matrix(omega: np.ndarray) -> np.ndarray:
    """
    Build the 4x4 Omega matrix for quaternion kinematics.

    q_dot = 0.5 * Omega(omega) * q

    Omega(ω) = | 0   -ωx  -ωy  -ωz |
               | ωx   0    ωz  -ωy |
               | ωy  -ωz   0    ωx |
               | ωz   ωy  -ωx   0  |
    """
    wx, wy, wz = omega
    return np.array([
        [0,   -wx, -wy, -wz],
        [wx,   0,   wz, -wy],
        [wy,  -wz,  0,   wx],
        [wz,   wy, -wx,  0 ]
    ])
