'''
Torch implementations adapted from the numpy implementations here:
https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py
'''
import numpy as np
import torch

def screwExp(expc6):
    # convert R6 to axis angle (w, v) and theta
    axisang, theta = AxisAng6(expc6)
    omg, nu = axisang[:3], axisang[3:]
    so3mat = vecToSo3(omg)

    # calculate the rotation exp([w] * theta)
    omgmat = vecToSo3(omg)
    exp3 = torch.eye(3) + torch.sin(theta)*omgmat + (1 - torch.cos(theta)) * torch.matmul(omgmat,omgmat)
    
    # calculate the translation K(S, theta)
    KStheta = (torch.eye(3)*theta + (1-torch.cos(theta))*so3mat + (theta-torch.sin(theta))*torch.matmul(so3mat, so3mat))
    KStheta = torch.matmul(KStheta, torch.unsqueeze(nu,1))

    # exponential SE3 exp([S] * theta)
    expStheta = torch.Tensor(4,4)
    expStheta[:3,:3] = exp3
    expStheta[:3,3] = KStheta.squeeze()
    expStheta[3,3] = 1
    return expStheta


def NearZero(z):
    """Determines whether a scalar is small enough to be treated as zero
    :param z: A scalar input to check
    :return: True if z is close to zero, false otherwise
    Example Input:
        z = -1e-7
    Output:
        True
    """
    return torch.abs(z) < 1e-6


def Normalize(V):
    """Normalizes a vector
    :param V: A vector
    :return: A unit vector pointing in the same direction as z
    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return V / torch.norm(V)


def AxisAng6(expc6):
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form
    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S
    Example Input:
        expc6 = np.array([1, 0, 0, 1, 2, 3])
    Output:
        (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
    """
    theta = torch.norm(expc6[:3])
    if NearZero(theta):
        theta = torch.norm(expc6[3:])
    return (expc6/theta, theta)


def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form
    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle
    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (Normalize(expc3), torch.norm(expc3))


def vecToSo3(omg):
    """Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
    Example Input:
        omg = np.array([1, 2, 3])
    Output:
        np.array([[ 0, -3,  2],
                  [ 3,  0, -1],
                  [-2,  1,  0]])
    """
    return torch.Tensor([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])


def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat
    Example Input:
        so3mat = np.array([[ 0, -3,  2],
                           [ 3,  0, -1],
                           [-2,  1,  0]])
    Output:
        np.array([1, 2, 3])
    """
    return torch.Tensor([so3mat[2][1], so3mat[0][2], so3mat[1][0]])