"""PyTorch implementation of the SO(3) RPY drone dynamics model. 

This module provides PyTorch-compatible dynamics functions for use with
MPPI and other gradient-based optimization methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from typing import Optional, Union


def quat_to_rotation_matrix(quat: torch. Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to rotation matrix. 
    
    Args:
        quat: Quaternion tensor of shape (..., 4) in xyzw format
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    x, y, z, w = torch.split(quat, 1, dim=-1)
    
    # Compute rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    
    R00 = 1 - 2 * (yy + zz)
    R01 = 2 * (xy - zw)
    R02 = 2 * (xz + yw)
    R10 = 2 * (xy + zw)
    R11 = 1 - 2 * (xx + zz)
    R12 = 2 * (yz - xw)
    R20 = 2 * (xz - yw)
    R21 = 2 * (yz + xw)
    R22 = 1 - 2 * (xx + yy)
    
    row1 = torch.cat([R00, R01, R02], dim=-1)
    row2 = torch.cat([R10, R11, R12], dim=-1)
    row3 = torch.cat([R20, R21, R22], dim=-1)
    
    rot_matrix = torch.stack([row1, row2, row3], dim=-2)
    
    return rot_matrix


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to Euler angles (xyz/roll-pitch-yaw).
    
    Args:
        quat:  Quaternion tensor of shape (... , 4) in xyzw format
        
    Returns: 
        Euler angles tensor of shape (..., 3) in xyz order
    """
    # Normalize quaternion
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)
    
    x, y, z, w = torch.split(quat, 1, dim=-1)
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    # Clamp to avoid numerical issues at poles
    sinp = torch.clamp(sinp, -1.0, 1.0)
    pitch = torch.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    return torch.cat([roll, pitch, yaw], dim=-1)


def euler_to_rotation_matrix(euler: torch. Tensor) -> torch.Tensor:
    """Convert Euler angles (xyz) to rotation matrix.
    
    Args:
        euler: Euler angles tensor of shape (..., 3) in xyz order
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    roll, pitch, yaw = torch.split(euler, 1, dim=-1)
    
    # Compute trig values
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    
    # Rotation matrix for ZYX Euler angles
    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr
    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr
    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr
    
    row1 = torch.cat([R00, R01, R02], dim=-1)
    row2 = torch.cat([R10, R11, R12], dim=-1)
    row3 = torch.cat([R20, R21, R22], dim=-1)
    
    rot_matrix = torch.stack([row1, row2, row3], dim=-2)
    
    return rot_matrix


def ang_vel_to_rpy_rates(quat: torch.Tensor, ang_vel: torch.Tensor) -> torch.Tensor:
    """Convert angular velocity to RPY rates.
    
    Args:
        quat: Quaternion tensor of shape (..., 4) in xyzw format
        ang_vel: Angular velocity tensor of shape (..., 3)
        
    Returns:
        RPY rates tensor of shape (... , 3)
    """
    euler = quat_to_euler(quat)
    roll, pitch, yaw = torch.split(euler, 1, dim=-1)
    
    p, q, r = torch.split(ang_vel, 1, dim=-1)
    
    # Conversion matrix from body rates to Euler rates
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    
    # Avoid singularity at pitch = ±90°
    cp = torch.clamp(cp, min=1e-6)
    
    roll_rate = p + sr * sp / cp * q + cr * sp / cp * r
    pitch_rate = cr * q - sr * r
    yaw_rate = sr / cp * q + cr / cp * r
    
    return torch.cat([roll_rate, pitch_rate, yaw_rate], dim=-1)


def rpy_rates_to_ang_vel(euler: torch.Tensor, rpy_rates: torch.Tensor) -> torch.Tensor:
    """Convert RPY rates to angular velocity.
    
    Args:
        euler: Euler angles tensor of shape (..., 3)
        rpy_rates: RPY rates tensor of shape (..., 3)
        
    Returns:
        Angular velocity tensor of shape (..., 3)
    """
    roll, pitch, yaw = torch. split(euler, 1, dim=-1)
    roll_rate, pitch_rate, yaw_rate = torch.split(rpy_rates, 1, dim=-1)
    
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cr = torch.cos(roll)
    sr = torch.sin(roll)
    
    # Conversion matrix from Euler rates to body rates
    p = roll_rate - sp * yaw_rate
    q = cr * pitch_rate + sr * cp * yaw_rate
    r = -sr * pitch_rate + cr * cp * yaw_rate
    
    return torch.cat([p, q, r], dim=-1)


def dynamics_euler(
    state: torch.Tensor,
    cmd: torch.Tensor,
    dt: float,
    mass: float,
    gravity_vec: torch.Tensor,
    J:  torch.Tensor,
    J_inv: torch.Tensor,
    acc_coef: torch.Tensor,
    cmd_f_coef: torch.Tensor,
    rpy_coef: torch.Tensor,
    rpy_rates_coef: torch.Tensor,
    cmd_rpy_coef: torch. Tensor,
    dist_f:  Optional[torch.Tensor] = None,
    dist_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fitted model with linear, second order RPY dynamics (Euler representation).
    
    This is the PyTorch version of the dynamics suitable for MPPI.
    State is in Euler angle representation for direct attitude control.
    
    Args:
        state: State tensor of shape (..., 12) containing:
            [pos (3), rpy (3), vel (3), drpy (3)]
        cmd: Command tensor of shape (... , 4) containing:
            [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        dt: Time step for integration (s)
        mass: Mass of the drone (kg)
        gravity_vec: Gravity vector (m/s^2), shape (3,)
        J: Inertia matrix (kg m^2), shape (3, 3)
        J_inv: Inverse inertia matrix, shape (3, 3)
        acc_coef: Coefficient for acceleration (1/s^2), scalar or shape (1,)
        cmd_f_coef: Coefficient for collective thrust (N/rad^2), scalar or shape (1,)
        rpy_coef: Coefficient for RPY dynamics (1/s), shape (3,)
        rpy_rates_coef: Coefficient for RPY rates dynamics (1/s^2), shape (3,)
        cmd_rpy_coef: Coefficient for RPY command dynamics (1/s), shape (3,)
        dist_f:  Disturbance force (N) in world frame, optional
        dist_t:  Disturbance torque (Nm) in world frame, optional
        
    Returns:
        Next state tensor of shape (..., 12)
    """
    # Extract state components
    pos = state[..., 0:3]
    rpy = state[..., 3:6]
    vel = state[..., 6:9]
    drpy = state[..., 9:12]
    
    # Extract command components
    cmd_rpy = cmd[..., 0:3]
    cmd_thrust = cmd[..., 3:4]
    
    # Compute rotation matrix from Euler angles
    rot = euler_to_rotation_matrix(rpy)
    
    # Compute thrust magnitude
    thrust = acc_coef + cmd_f_coef * cmd_thrust
    
    # Thrust vector in body frame (z-axis)
    thrust_body = torch.zeros_like(vel)
    thrust_body[..., 2] = thrust. squeeze(-1)
    
    # Transform thrust to world frame
    thrust_world = torch.matmul(rot, thrust_body. unsqueeze(-1)).squeeze(-1)
    
    # Linear dynamics
    pos_dot = vel
    vel_dot = thrust_world / mass + gravity_vec
    
    if dist_f is not None: 
        vel_dot = vel_dot + dist_f / mass
    
    # Rotational dynamics (second-order linear model)
    rpy_dot = drpy
    ddrpy = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy
    
    # If torque disturbances are present, convert to RPY acceleration
    if dist_t is not None:
        # Convert current angular acceleration to torque
        ang_vel = rpy_rates_to_ang_vel(rpy, drpy)
        
        # Compute torque from angular acceleration
        # τ = J * α + ω × (J * ω)
        torque = torch.matmul(J, ddrpy. unsqueeze(-1)).squeeze(-1)
        torque = torque + torch.cross(
            ang_vel,
            torch.matmul(J, ang_vel.unsqueeze(-1)).squeeze(-1),
            dim=-1
        )
        
        # Transform disturbance torque to body frame and add
        rot_inv = rot.transpose(-2, -1)
        dist_t_body = torch.matmul(rot_inv, dist_t.unsqueeze(-1)).squeeze(-1)
        torque = torque + dist_t_body
        
        # Convert back to angular acceleration
        # Remove gyroscopic term
        torque = torque - torch.cross(
            ang_vel,
            torch.matmul(J, ang_vel.unsqueeze(-1)).squeeze(-1),
            dim=-1
        )
        ddrpy = torch.matmul(J_inv, torque.unsqueeze(-1)).squeeze(-1)
    
    # Integrate state
    pos_next = pos + pos_dot * dt
    vel_next = vel + vel_dot * dt
    rpy_next = rpy + rpy_dot * dt
    drpy_next = drpy + ddrpy * dt
    
    # Wrap angles to [-pi, pi]
    rpy_next = torch. atan2(torch.sin(rpy_next), torch.cos(rpy_next))
    
    # Concatenate next state
    state_next = torch.cat([pos_next, rpy_next, vel_next, drpy_next], dim=-1)
    
    return state_next


def dynamics_quat(
    state: torch. Tensor,
    cmd: torch.Tensor,
    dt: float,
    mass: float,
    gravity_vec: torch. Tensor,
    J: torch.Tensor,
    J_inv: torch.Tensor,
    acc_coef: torch. Tensor,
    cmd_f_coef: torch.Tensor,
    rpy_coef: torch.Tensor,
    rpy_rates_coef:  torch.Tensor,
    cmd_rpy_coef: torch.Tensor,
    dist_f: Optional[torch.Tensor] = None,
    dist_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fitted model with quaternion state representation. 
    
    This version uses quaternions for the state but still applies the
    second-order linear RPY dynamics model.
    
    Args:
        state: State tensor of shape (..., 13) containing:
            [pos (3), quat (4, xyzw), vel (3), ang_vel (3)]
        cmd: Command tensor of shape (..., 4) containing:
            [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        dt:  Time step for integration (s)
        (other args same as dynamics_euler)
        
    Returns:
        Next state tensor of shape (..., 13)
    """
    # Extract state components
    pos = state[..., 0:3]
    quat = state[..., 3:7]
    vel = state[..., 7:10]
    ang_vel = state[..., 10:13]
    
    # Convert to Euler representation
    rpy = quat_to_euler(quat)
    drpy = ang_vel_to_rpy_rates(quat, ang_vel)
    
    # Extract command components
    cmd_rpy = cmd[..., 0:3]
    cmd_thrust = cmd[..., 3:4]
    
    # Compute rotation matrix
    rot = quat_to_rotation_matrix(quat)
    
    # Compute thrust magnitude
    thrust = acc_coef + cmd_f_coef * cmd_thrust
    
    # Thrust vector in body frame (z-axis)
    thrust_body = torch.zeros_like(vel)
    thrust_body[..., 2] = thrust.squeeze(-1)
    
    # Transform thrust to world frame
    thrust_world = torch.matmul(rot, thrust_body.unsqueeze(-1)).squeeze(-1)
    
    # Linear dynamics
    pos_dot = vel
    vel_dot = thrust_world / mass + gravity_vec
    
    if dist_f is not None:
        vel_dot = vel_dot + dist_f / mass
    
    # Rotational dynamics using second-order RPY model
    ddrpy = rpy_coef * rpy + rpy_rates_coef * drpy + cmd_rpy_coef * cmd_rpy
    
    # Convert RPY acceleration to angular acceleration
    # This is an approximation; for exact conversion, we'd need the Jacobian
    ang_vel_dot = rpy_rates_to_ang_vel(rpy, ddrpy)
    
    if dist_t is not None: 
        # Add torque disturbances
        torque = torch.matmul(J, ang_vel_dot.unsqueeze(-1)).squeeze(-1)
        torque = torque + torch.cross(
            ang_vel,
            torch.matmul(J, ang_vel. unsqueeze(-1)).squeeze(-1),
            dim=-1
        )
        
        # Transform disturbance torque to body frame
        rot_inv = rot. transpose(-2, -1)
        dist_t_body = torch.matmul(rot_inv, dist_t.unsqueeze(-1)).squeeze(-1)
        torque = torque + dist_t_body
        
        # Convert back to angular acceleration
        torque = torque - torch.cross(
            ang_vel,
            torch.matmul(J, ang_vel.unsqueeze(-1)).squeeze(-1),
            dim=-1
        )
        ang_vel_dot = torch.matmul(J_inv, torque.unsqueeze(-1)).squeeze(-1)
    
    # Quaternion derivative
    # q_dot = 0.5 * Ω(ω) * q
    wx, wy, wz = torch.split(ang_vel, 1, dim=-1)
    zero = torch.zeros_like(wx)
    
    # Skew-symmetric matrix extended for quaternion kinematics
    omega_matrix = torch.cat([
        torch.cat([zero, -wx, -wy, -wz], dim=-1).unsqueeze(-2),
        torch.cat([wx, zero, wz, -wy], dim=-1).unsqueeze(-2),
        torch.cat([wy, -wz, zero, wx], dim=-1).unsqueeze(-2),
        torch.cat([wz, wy, -wx, zero], dim=-1).unsqueeze(-2),
    ], dim=-2)
    
    quat_dot = 0.5 * torch.matmul(omega_matrix, quat. unsqueeze(-1)).squeeze(-1)
    
    # Integrate state
    pos_next = pos + pos_dot * dt
    vel_next = vel + vel_dot * dt
    quat_next = quat + quat_dot * dt
    ang_vel_next = ang_vel + ang_vel_dot * dt
    
    # Normalize quaternion
    quat_next = quat_next / torch.norm(quat_next, dim=-1, keepdim=True)
    
    # Concatenate next state
    state_next = torch.cat([pos_next, quat_next, vel_next, ang_vel_next], dim=-1)
    
    return state_next


class DroneModelTorch: 
    """PyTorch wrapper for drone dynamics with parameter management."""
    
    def __init__(
        self,
        parameters: dict,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float64,
        use_euler_state: bool = True,
    ):
        """Initialize the PyTorch drone model.
        
        Args:
            parameters: Dictionary containing model parameters
            device: Device to place tensors on
            dtype: Data type for tensors
            use_euler_state:  If True, use Euler state (12D), else use quaternion state (13D)
        """
        self.device = device
        self.dtype = dtype
        self.use_euler_state = use_euler_state
        
        # Convert parameters to tensors
        self. mass = float(parameters["mass"])
        self.gravity_vec = torch.tensor(
            parameters["gravity_vec"], dtype=dtype, device=device
        )
        self.J = torch.tensor(parameters["J"], dtype=dtype, device=device)
        self.J_inv = torch.tensor(parameters["J_inv"], dtype=dtype, device=device)
        self.acc_coef = torch. tensor(
            parameters["acc_coef"], dtype=dtype, device=device
        )
        self.cmd_f_coef = torch.tensor(
            parameters["cmd_f_coef"], dtype=dtype, device=device
        )
        self.rpy_coef = torch.tensor(
            parameters["rpy_coef"], dtype=dtype, device=device
        )
        self.rpy_rates_coef = torch.tensor(
            parameters["rpy_rates_coef"], dtype=dtype, device=device
        )
        self.cmd_rpy_coef = torch.tensor(
            parameters["cmd_rpy_coef"], dtype=dtype, device=device
        )
        
        # Control limits
        self.thrust_min = parameters. get("thrust_min", 0.0)
        self.thrust_max = parameters.get("thrust_max", 1.0)
        self.rpy_max = parameters.get("rpy_max", 0.5)  # radians
        
        # State dimensions
        self.nx = 12 if use_euler_state else 13
        self.nu = 4  # [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
    
    def dynamics(
        self,
        state: torch.Tensor,
        cmd: torch.Tensor,
        dt: float,
        dist_f: Optional[torch. Tensor] = None,
        dist_t: Optional[torch. Tensor] = None,
    ) -> torch.Tensor:
        """Compute next state given current state and command.
        
        Args:
            state: Current state
            cmd: Control command
            dt: Time step
            dist_f: Force disturbance (optional)
            dist_t: Torque disturbance (optional)
            
        Returns:
            Next state
        """
        if self.use_euler_state:
            return dynamics_euler(
                state, cmd, dt,
                self.mass, self. gravity_vec, self.J, self.J_inv,
                self.acc_coef, self.cmd_f_coef, self.rpy_coef,
                self.rpy_rates_coef, self.cmd_rpy_coef,
                dist_f, dist_t
            )
        else:
            return dynamics_quat(
                state, cmd, dt,
                self.mass, self.gravity_vec, self.J, self. J_inv,
                self. acc_coef, self.cmd_f_coef, self. rpy_coef,
                self.rpy_rates_coef, self.cmd_rpy_coef,
                dist_f, dist_t
            )
    
    def obs_to_state(self, obs: dict) -> torch.Tensor:
        """Convert observation dictionary to state tensor. 
        
        Args:
            obs: Observation dictionary from environment
            
        Returns:
            State tensor
        """
        if self.use_euler_state:
            # Convert quaternion to Euler angles
            quat = obs["quat"]
            rot = R.from_quat(quat)
            rpy = rot. as_euler("xyz")
            
            # Convert angular velocity to RPY rates
            from drone_models.utils. rotation import ang_vel2rpy_rates
            drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
            
            state = np.concatenate([
                obs["pos"],
                rpy,
                obs["vel"],
                drpy
            ])
        else:
            state = np.concatenate([
                obs["pos"],
                obs["quat"],
                obs["vel"],
                obs["ang_vel"]
            ])
        
        return torch.tensor(state, dtype=self.dtype, device=self.device)
