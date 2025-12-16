"""Improved pure MPPI controller for quadrotor attitude control. 

This module implements a robust MPPI controller that directly outputs
roll-pitch-yaw and thrust commands with better cost shaping and dynamics. 
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Import PyTorch MPPI
import pytorch_mppi

# Import our PyTorch dynamics model
from drone_models.core import load_params
from lsy_drone_racing.utils.model_torch import DroneModelTorch

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeMPPIController(Controller):
    """Improved pure MPPI controller for attitude control (RPY + Thrust)."""

    def __init__(
        self,
        obs:  dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
    ):
        """Initialize the MPPI attitude controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)

        # Simulation parameters
        self._dt = 1 / config.env.freq
        self._config = config

        # MPPI hyperparameters - optimized for robust trajectory optimization
        self.mppi_horizon = 28  # Planning horizon steps (increased slightly for better foresight)
        self.mppi_dt = self._dt * 2  # 0.04s time discretization
        self.num_samples = 6000  # Stable optimization quality
        self.lambda_weight = 8.5  # Temperature parameter (slightly lower for more exploration)
        
        # Gate geometry constants
        self.gate_opening = 0.30  # 30cm square opening
        self.gate_frame_thickness = 0.045  # 4.5cm frame

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        print(f"[AttitudeMPPI] Initializing on device: {self.device}")

        # Load drone parameters
        self.drone_params = load_params("so_rpy", config.sim.drone_model)

        # Create PyTorch dynamics model
        self.drone_model = DroneModelTorch(
            parameters=self.drone_params,
            device=self.device,
            dtype=self.dtype,
            use_euler_state=True,
        )

        # Extract gate information
        self.gates_pos = []
        self.gates_quat = []
        self.gates_normal = []

        for gate in config.env.track.gates:
            pos = np.array(gate["pos"], dtype=float)
            rpy = np.array(gate["rpy"], dtype=float)
            rot = R.from_euler("xyz", rpy)

            self.gates_pos.append(pos)
            self.gates_quat.append(rot.as_quat())
            self.gates_normal.append(rot.as_matrix()[:, 0])

        self.gates_pos = np.array(self.gates_pos)
        self.gates_quat = np.array(self.gates_quat)
        self.gates_normal = np.array(self.gates_normal)

        # Current target
        self.target_gate_idx = int(obs["target_gate"])
        self.goal = self.gates_pos[self.target_gate_idx]
        self.prev_goal = self.goal.copy()

        # More conservative control limits
        self.rpy_max = 0.3  # ±17 degrees (more conservative)
        self.thrust_min = self.drone_params["thrust_min"] * 4
        self.thrust_max = self.drone_params["thrust_max"] * 4
        self.hover_thrust = self.drone_params["mass"] * abs(self.drone_params["gravity_vec"][-1])

        # Simplified cost weights - balanced aggressive for 4-gate success
        self.cost_weights = {
              "position": torch.tensor([19.0, 19.0, 15.5], device=self.device, dtype=self.dtype),  # Slightly reduced for smoother approach
              "velocity": torch.tensor([0.038, 0.038, 0.120], device=self.device, dtype=self.dtype),  # Slightly reduced damping
            "attitude": torch.tensor([1.4, 1.4, 0.18], device=self.device, dtype=self.dtype),  # Slightly lower for more agility
            "z_floor": 2000.0,  # Strong penalty for ground collision (Z < 0.03m)
        }

        # Store previous control for rate penalty
        self.prev_control = None

        # Define dynamics wrapper for MPPI
        def dynamics_fn(state, control):
            """Dynamics function for MPPI."""
            return self.drone_model.dynamics(state, control, self.mppi_dt)

        # Define running cost for MPPI
        def running_cost_fn(state, control):
            """Running cost for MPPI trajectory optimization."""
            return self.compute_running_cost(state, control)

        # Control bounds - more conservative
        u_min = torch.tensor(
            [-self.rpy_max, -self.rpy_max, -self.rpy_max, self.thrust_min],
            dtype=self.dtype,
            device=self.device
        )
        u_max = torch.tensor(
            [self.rpy_max, self.rpy_max, self.rpy_max, self.thrust_max],
            dtype=self.dtype,
            device=self.device
        )

        # Smaller control noise for smoother sampling
        noise_sigma = torch.diag(torch.tensor(
            [0.05, 0.05, 0.05, 0.2],  # Reduced noise
            dtype=self.dtype,
            device=self.device
        ))

        # Initialize MPPI solver
        self.mppi = pytorch_mppi.MPPI(
            dynamics=dynamics_fn,
            running_cost=running_cost_fn,
            nx=self.drone_model.nx,
            noise_sigma=noise_sigma,
            num_samples=self.num_samples,
            horizon=self.mppi_horizon,
            lambda_=self.lambda_weight,
            u_min=u_min,
            u_max=u_max,
            device=self.device,
        )

        # Initialize control sequence to hover
        initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
        initial_control[: , 3] = self.hover_thrust
        self.mppi.U = initial_control

        self.step_count = 0

        print("[AttitudeMPPI] Initialization complete - Conservative tuning for stability")
        print(f"  - Horizon: {self.mppi_horizon} steps @ {self.mppi_dt:.3f}s = {self.mppi_horizon * self.mppi_dt:.2f}s")
        print(f"  - Samples: {self.num_samples}")
        print(f"  - Lambda: {self.lambda_weight}")
        print(f"  - RPY limits: ±{np.rad2deg(self.rpy_max):.1f}°")
        print(f"  - Gate opening: {self.gate_opening*100:.0f}cm square (frame: {self.gate_frame_thickness*100:.1f}cm)")

    def compute_running_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """Simplified MPPI running cost - removed redundant terms.

        Args:
            state: State tensor (..., 12): [pos(3), rpy(3), vel(3), drpy(3)]
            control: Control tensor (..., 4): [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]

        Returns:
            Cost tensor of shape (...)
        """
        # Extract state components
        pos = state[..., 0:3]
        rpy = state[..., 3:6]
        vel = state[..., 6:9]

        # 1. Position tracking - MAIN OBJECTIVE with distance-adaptive scaling
        goal_t = torch.tensor(self.goal, dtype=self.dtype, device=self.device)
        pos_error = pos - goal_t
        dist_to_goal = torch.norm(pos_error, dim=-1)
        # 2.1x position cost when within 0.4m for tighter precision at gate (slightly reduced)
        proximity_scale = torch.where(dist_to_goal < 0.4, 2.1, 1.0)
        c_pos = proximity_scale * torch.sum(self.cost_weights["position"] * pos_error ** 2, dim=-1)

        # 1a. Attraction to gate opening and obstacle avoidance for gate frames
        # Transform position into gate-local coordinates: [x_normal, y_plane, z_plane]
        gate_center_np = self.gates_pos[self.target_gate_idx]
        gate_quat_np = self.gates_quat[self.target_gate_idx]
        gate_R_np = R.from_quat(gate_quat_np).as_matrix()
        gate_R = torch.tensor(gate_R_np, dtype=self.dtype, device=self.device)
        gate_center = torch.tensor(gate_center_np, dtype=self.dtype, device=self.device)
        # local coordinates
        rel = pos - gate_center
        x_n = torch.sum(rel * gate_R[:, 0], dim=-1)
        y_p = torch.sum(rel * gate_R[:, 1], dim=-1)
        z_p = torch.sum(rel * gate_R[:, 2], dim=-1)

        # Opening and frame geometry
        opening_half = self.gate_opening / 2.0  # 0.15 m
        frame_t = self.gate_frame_thickness      # 0.045 m
        edge_offset = opening_half + frame_t * 0.5
        safety_r = 0.05  # keep ~5cm away from frames

        # Attraction: encourage being inside opening box in gate plane
        # Hinge loss if outside opening in y or z
        y_excess = torch.clamp(torch.abs(y_p) - opening_half, min=0.0)
        z_excess = torch.clamp(torch.abs(z_p) - opening_half, min=0.0)
        w_open = 11.5  # Slightly reduced for smoother correction
        c_open_hinge = w_open * (y_excess ** 2 + z_excess ** 2)

        # Soft attraction toward the 2D opening center (y=0, z=0) when near the gate plane
        near_plane = torch.abs(x_n) < 0.6
        w_center = 4.3  # Slightly reduced for smoother approach
        c_center = torch.where(near_plane, w_center * (y_p ** 2 + z_p ** 2), torch.zeros_like(x_n))

        # Obstacle avoidance: vertical frames modeled as infinite poles at y=±edge_offset
        # Penalize inverse-square proximity inside safety radius
        dy_left = torch.abs(y_p + edge_offset)
        dy_right = torch.abs(y_p - edge_offset)
        w_vert = 120.0
        c_vert = w_vert * (torch.clamp(safety_r - dy_left, min=0.0) ** 2 + torch.clamp(safety_r - dy_right, min=0.0) ** 2)

        # Obstacle avoidance: horizontal frames modeled as capsules along y at z=±edge_offset
        dz_top = torch.abs(z_p - edge_offset)
        dz_bottom = torch.abs(z_p + edge_offset)
        # Active only when within lateral span around the gate opening (|y| <= opening_half + frame_t)
        within_span = torch.abs(y_p) <= (opening_half + frame_t)
        w_horiz = 100.0
        c_horiz = torch.where(within_span, w_horiz * (torch.clamp(safety_r - dz_top, min=0.0) ** 2 + torch.clamp(safety_r - dz_bottom, min=0.0) ** 2), torch.zeros_like(dz_top))

        c_gate_struct = c_open_hinge + c_center + c_vert + c_horiz

        # 2. Velocity penalty - light damping
        c_vel = torch.sum(self.cost_weights["velocity"] * vel ** 2, dim=-1)

        # 2a. Risk-aware slowdown: if too close to any frame, add speed penalty
        min_frame_dist = torch.minimum(
            torch.minimum(dy_left, dy_right),
            torch.minimum(dz_top, dz_bottom)
        )
        high_risk = min_frame_dist < (safety_r + 0.01)
        w_slow = 0.4
        c_slow = torch.where(high_risk, w_slow * torch.sum(vel ** 2, dim=-1), torch.zeros_like(min_frame_dist))

        # 3. Attitude - prefer level flight
        c_att = torch.sum(self.cost_weights["attitude"] * rpy ** 2, dim=-1)

        # 4. Ground collision penalty (critical safety constraint)
        z_violation = torch.clamp(0.03 - pos[..., 2], min=0.0)
        c_z_floor = self.cost_weights["z_floor"] * z_violation ** 2

        # Total cost (simplified) + gate structure shaping and risk-aware slowdown
        total_cost = c_pos + c_vel + c_att + c_z_floor + c_gate_struct + c_slow

        return total_cost

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control command using MPPI. 

        Args:
            obs:  Current observation
            info: Additional info (optional)

        Returns:
            Control command:  [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        """
        self.step_count += 1

        # Update target gate if necessary
        new_target_idx = int(obs["target_gate"])
        if new_target_idx != self.target_gate_idx:
            self.target_gate_idx = new_target_idx
            self.prev_goal = self.goal.copy()
            self.goal = self.gates_pos[self.target_gate_idx]
            print(f"\n[AttitudeMPPI] Updated target to gate {self.target_gate_idx}")
            
            # Optionally reset control sequence on gate change
            initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
            initial_control[:, 3] = self.hover_thrust
            self.mppi.U = initial_control

        # Convert observation to state tensor
        state = self.drone_model.obs_to_state(obs)

        # Run MPPI optimization
        with torch.no_grad():
            optimal_control = self.mppi.command(state)

        # Store for rate penalty
        self.prev_control = optimal_control.clone()

        # Convert to numpy
        control_np = optimal_control.cpu().numpy()

        # Apply additional safety clipping
        control_np[0: 3] = np.clip(control_np[0:3], -self.rpy_max, self.rpy_max)
        control_np[3] = np.clip(control_np[3], self.thrust_min, self.thrust_max)

        # Enhanced debug output
        if self.step_count % 10 == 0:
            pos = obs["pos"]
            vel = obs["vel"]
            gate_pos = self.gates_pos[self.target_gate_idx]
            dist_to_gate = np.linalg.norm(pos - gate_pos)
            z_offset = pos[2] - gate_pos[2]
            xy_dist = np.linalg.norm(pos[:2] - gate_pos[:2])
            speed = np.linalg.norm(vel)
            print(f"[AttitudeMPPI] Step {self.step_count:4d} | "
                  f"G{self.target_gate_idx} | "
                  f"Pos:[{pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:5.2f}] | "
                  f"3D:{dist_to_gate:4.2f}m XY:{xy_dist:4.2f}m ΔZ:{z_offset:+.2f}m | "
                  f"V:{speed:.1f}m/s | "
                  f"T:{control_np[3]:.2f}N")

        return control_np

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info:  dict,
    ) -> bool:
        """Step callback."""
        return False

    def episode_callback(self):
        """Episode callback to reset state."""
        # Reset control sequence
        initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
        initial_control[:, 3] = self.hover_thrust
        self.mppi.U = initial_control
        
        # Reset step counter and previous control
        self.step_count = 0
        self.prev_control = None
        
        print("[AttitudeMPPI] Episode reset")
