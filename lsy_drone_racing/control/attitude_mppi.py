"""This module implements an MPPI (Model Predictive Path Integral) controller for quadrotor control.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints. The MPPI algorithm samples control perturbations
around a baseline trajectory and weights them based on their cost to compute optimal controls.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
"""

from __future__ import annotations  # Python 3.10 type hints

import math
from typing import TYPE_CHECKING

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeMPPI(Controller):
    """MPPI controller using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the MPPI attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._dt = 1.0 / self._freq

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = drone_params["mass"]
        self.g = 9.81

        # PID gains for baseline controller
        self.kp = np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4])
        self.ki_range = np.array([2.0, 2.0, 0.4])
        self.i_error = np.zeros(3)

        # MPPI parameters
        self.horizon = 10  # Number of timesteps to look ahead
        self.num_samples = 100  # Number of trajectory samples
        self.lambda_ = 1.0  # Temperature parameter for exponential weighting
        self.noise_std = np.array([0.1, 0.1, 0.1, 5.0])  # Noise std for [roll, pitch, yaw, thrust]

        # Same waypoints as in the other controllers
        waypoints = np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )
        self._t_total = 15  # s
        t = np.linspace(0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._des_vel_spline = self._des_pos_spline.derivative()
        self._des_acc_spline = self._des_vel_spline.derivative()

        # Initialize control sequence (baseline trajectory)
        self.control_sequence = np.zeros((self.horizon, 4))  # [roll, pitch, yaw, thrust]
        for i in range(self.horizon):
            self.control_sequence[i] = np.array([0.0, 0.0, 0.0, self.drone_mass * self.g])

        self._tick = 0
        self._finished = False

    def _compute_baseline_control(
        self, obs: dict[str, NDArray[np.floating]], t: float
    ) -> NDArray[np.floating]:
        """Compute baseline control using PID to track reference trajectory.

        Args:
            obs: Current observation.
            t: Current time.

        Returns:
            Baseline control [roll, pitch, yaw, thrust].
        """
        # Get reference trajectory from spline
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)
        des_acc = self._des_acc_spline(t)
        des_yaw = 0.0

        # Calculate the deviations from the desired trajectory
        pos_error = des_pos - obs["pos"]
        vel_error = des_vel - obs["vel"]

        # Update integral error
        self.i_error += pos_error * self._dt
        self.i_error = np.clip(self.i_error, -self.ki_range, self.ki_range)

        # Compute target thrust vector using PID + feedforward
        target_thrust = np.zeros(3)
        target_thrust += self.kp * pos_error
        target_thrust += self.ki * self.i_error
        target_thrust += self.kd * vel_error
        target_thrust += self.drone_mass * des_acc  # Feedforward from reference acceleration
        target_thrust[2] += self.drone_mass * self.g

        # Compute desired orientation from target thrust
        z_axis_desired = target_thrust / np.linalg.norm(target_thrust)
        x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_axis_desired = np.cross(z_axis_desired, x_c_des)
        y_axis_desired /= np.linalg.norm(y_axis_desired)
        x_axis_desired = np.cross(y_axis_desired, z_axis_desired)

        R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
        euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)

        # Compute desired thrust magnitude
        z_axis = R.from_quat(obs["quat"]).as_matrix()[:, 2]
        thrust_desired = target_thrust.dot(z_axis)

        return np.array([euler_desired[0], euler_desired[1], euler_desired[2], thrust_desired])

    def _update_baseline_sequence(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Update the baseline control sequence to follow the reference trajectory.

        For each timestep in the horizon, compute the control that would track the reference
        trajectory at that future time. This provides a good baseline for MPPI perturbations.

        Args:
            obs: Current observation.
        """
        # Compute baseline control for the horizon using reference trajectory
        # We use the current obs as a starting point and compute controls for future reference points
        for i in range(self.horizon):
            t_future = min((self._tick + i) / self._freq, self._t_total)
            
            # For future timesteps, we compute what control would be needed to track
            # the reference at that time, assuming we follow the reference trajectory
            if i == 0:
                # For the current timestep, use actual observation
                self.control_sequence[i] = self._compute_baseline_control(obs, t_future)
            else:
                # For future timesteps, create a "virtual" observation at the reference position
                # This is a simplification that assumes we'll be on the reference trajectory
                des_pos = self._des_pos_spline(t_future)
                des_vel = self._des_vel_spline(t_future)
                des_acc = self._des_acc_spline(t_future)
                des_yaw = 0.0
                
                # Compute feedforward control based on reference
                target_thrust = self.drone_mass * des_acc
                target_thrust[2] += self.drone_mass * self.g
                
                # Compute desired orientation from target thrust
                z_axis_desired = target_thrust / (np.linalg.norm(target_thrust) + 1e-6)
                x_c_des = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
                y_axis_desired = np.cross(z_axis_desired, x_c_des)
                y_axis_desired /= (np.linalg.norm(y_axis_desired) + 1e-6)
                x_axis_desired = np.cross(y_axis_desired, z_axis_desired)
                
                R_desired = np.vstack([x_axis_desired, y_axis_desired, z_axis_desired]).T
                euler_desired = R.from_matrix(R_desired).as_euler("xyz", degrees=False)
                thrust_desired = np.linalg.norm(target_thrust)
                
                self.control_sequence[i] = np.array([
                    euler_desired[0], euler_desired[1], euler_desired[2], thrust_desired
                ])

    def _evaluate_cost(
        self,
        obs: dict[str, NDArray[np.floating]],
        control: NDArray[np.floating],
        t: float,
    ) -> float:
        """Evaluate cost for a single control action.

        This is a simplified cost that penalizes deviation from reference and control effort.

        Args:
            obs: Current observation.
            control: Control to evaluate [roll, pitch, yaw, thrust].
            t: Current time.

        Returns:
            Cost value.
        """
        # Get reference state
        des_pos = self._des_pos_spline(t)
        des_vel = self._des_vel_spline(t)

        # Position and velocity tracking cost
        pos_cost = 50.0 * np.sum((obs["pos"] - des_pos) ** 2)
        vel_cost = 10.0 * np.sum((obs["vel"] - des_vel) ** 2)

        # Control effort cost (deviation from baseline hover)
        hover_control = np.array([0.0, 0.0, 0.0, self.drone_mass * self.g])
        control_cost = 1.0 * np.sum((control - hover_control) ** 2)

        return pos_cost + vel_cost + control_cost

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw using MPPI.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        # Update baseline control sequence to follow reference trajectory
        self._update_baseline_sequence(obs)

        # MPPI: Sample control perturbations around the baseline
        control_samples = np.zeros((self.num_samples, self.horizon, 4))
        costs = np.zeros(self.num_samples)

        for i in range(self.num_samples):
            # Generate noise perturbations
            noise = np.random.randn(self.horizon, 4) * self.noise_std

            # Add noise to baseline control sequence
            control_samples[i] = self.control_sequence + noise

            # Clip controls to reasonable bounds
            control_samples[i, :, 0:3] = np.clip(
                control_samples[i, :, 0:3], -0.5, 0.5
            )  # Clip roll, pitch, yaw
            control_samples[i, :, 3] = np.clip(
                control_samples[i, :, 3], 0.0, 4 * self.drone_mass * self.g
            )  # Clip thrust

            # Evaluate cost for this trajectory (simplified - just evaluate first step)
            t_eval = min((self._tick) / self._freq, self._t_total)
            costs[i] = self._evaluate_cost(obs, control_samples[i, 0], t_eval)

        # Compute weights using exponential transformation
        min_cost = np.min(costs)
        weights = np.exp(-1.0 / self.lambda_ * (costs - min_cost))
        weights /= np.sum(weights)

        # Compute weighted average of control sequences
        optimal_control_sequence = np.zeros((self.horizon, 4))
        for i in range(self.num_samples):
            optimal_control_sequence += weights[i] * control_samples[i]

        # Update control sequence for next iteration (shift and append)
        self.control_sequence[:-1] = optimal_control_sequence[1:]
        self.control_sequence[-1] = optimal_control_sequence[
            -1
        ]  # Repeat last control for final horizon step

        # Return the first control in the optimal sequence
        action = optimal_control_sequence[0].astype(np.float32)

        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Reset the internal state."""
        self.i_error[:] = 0
        self._tick = 0
        self._finished = False
        # Reset control sequence
        for i in range(self.horizon):
            self.control_sequence[i] = np.array([0.0, 0.0, 0.0, self.drone_mass * self.g])
