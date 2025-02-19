import torch
import numpy as np

def quat_to_euler(q: torch.Tensor, device: str) -> torch.Tensor:
    """Convert quaternion to Euler angles (roll, pitch, yaw)
    Args:
        q: quaternion tensor [w, x, y, z]
        device: torch device to use
    Returns:
        torch.Tensor: [roll, pitch, yaw]
    """
    # Extract quaternion components
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.tensor(3.14159 / 2.0, device=device),
        torch.asin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw])

def l2_dist(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1), np.square(vec2)))

def denormalize(val, min, max):
    return ((val + 1) / 2) * (max - min) + min

def get_DCM(phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    phi = torch.as_tensor(phi, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    psi = torch.as_tensor(psi, dtype=torch.float32)
    
    c_phi = torch.cos(phi)
    s_phi = torch.sin(phi)
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    c_psi = torch.cos(psi)
    s_psi = torch.sin(psi)
    
    m11 = c_theta * c_psi
    m12 = s_phi * s_theta * c_psi - c_phi * s_psi
    m13 = c_phi * s_theta * c_psi + s_phi * s_psi
    
    m21 = c_theta * s_psi
    m22 = s_phi * s_theta * s_psi + c_phi * c_psi
    m23 = c_phi * s_theta * s_psi - s_phi * c_psi
    
    m31 = -s_theta
    m32 = s_phi * c_theta
    m33 = c_phi * c_theta
    
    row1 = torch.stack([m11, m12, m13], dim=-1)
    row2 = torch.stack([m21, m22, m23], dim=-1)
    row3 = torch.stack([m31, m32, m33], dim=-1)
    
    L_EB = torch.stack([row1, row2, row3], dim=-2)
    
    return L_EB

def normalize(val, min, max):
    return 2 * (val - min) / (max - min) - 1

class AttrDict(dict):
    """Dictionary subclass that allows attribute access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_dict(data):
        """Recursively convert a dictionary into AttrDict."""
        if isinstance(data, dict):
            return AttrDict({key: AttrDict.from_dict(value) for key, value in data.items()})
        elif isinstance(data, list):
            return [AttrDict.from_dict(item) for item in data]
        else:
            return data