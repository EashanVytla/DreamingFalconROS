#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
import scipy
import numpy as np
from enum import Enum, auto
import math
from tqdm import tqdm
import yaml
from utils import AttrDict

class DroneState(Enum):
    ARMING = auto()
    TAKEOFF = auto()
    CHIRP = auto()
    LAND = auto()

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_chirp')

        with open('config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)

        config = AttrDict.from_dict(config_dict)

        self._mass = config.physics.mass #kg
        self._g = config.physics.g

        # State management
        self.current_state = DroneState.ARMING
        self.target_takeoff_height = -50.0  # Target height for takeoff
        self.takeoff_height_threshold = -4.8  # Height at which takeoff is considered complete
        self.landing_height_threshold = -0.2  # Height at which landing is considered complete

        # Chirp configuration
        self.chirp_x = 5 * scipy.signal.chirp(t=np.arange(0, 10000, 0.1), f0=0.1, t1=1000, f1=7, method="quadratic")
        self.chirp_y = 5 * scipy.signal.chirp(t=np.arange(0, 10000, 0.1), f0=0.2, t1=1000, f1=9, method="quadratic")
        self.chirp_z = 5 * scipy.signal.chirp(t=np.arange(0, 10000, 0.1), f0=0.3, t1=1000, f1=10, method="quadratic") - 18
        self.chirp_yaw = math.pi/2 * scipy.signal.chirp(t=np.arange(0, 10000, 0.1), f0=0.4, t1=1000, f1=6, method="quadratic")
        self.chirp_counter = 0
        self.pbar = None  # Initialize progress bar variable

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.takeoff_height = -5.0

        self.publish_counter = 0

        # Create a timer to publish control commands
        self.timer = self.create_timer(config.physics.refresh_rate, self.timer_callback)

    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode", throttle_duration_sec=1)

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = True
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw=1.57):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.yaw = yaw  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}", throttle_duration_sec=1)

    def publish_rate_setpoint(self, x: float, y: float, z: float, yaw: float):
        msg = TrajectorySetpoint()
        msg.acceleration = [x, y, z]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def handle_arming_state(self):
        """Handle the ARMING state behavior"""
        if self.offboard_setpoint_counter == 20:
            self.engage_offboard_mode()
            self.arm()
            self.current_state = DroneState.TAKEOFF
        
        if self.offboard_setpoint_counter < 21:
            self.publish_position_setpoint(0.0, 0.0, self.target_takeoff_height)
            self.offboard_setpoint_counter += 1

    def handle_takeoff_state(self):
        """Handle the TAKEOFF state behavior"""
        self.publish_position_setpoint(0.0, 0.0, self.target_takeoff_height)
        
        if abs(self.vehicle_local_position.z - self.target_takeoff_height) < 0.5:
            self.get_logger().info("Takeoff complete, starting chirp")
            self.current_state = DroneState.CHIRP
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)
            self.pbar = tqdm(total=len(self.chirp_x), desc="Chirp Progress")

    def handle_chirp_state(self):
        """Handle the CHIRP state behavior"""
        if self.chirp_counter >= len(self.chirp_x):
            self.pbar.close()
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
            self.current_state = DroneState.LAND
            return
            
        self.publish_rate_setpoint(
            self.chirp_x[self.chirp_counter],
            self.chirp_y[self.chirp_counter],
            self.chirp_z[self.chirp_counter],
            self.chirp_yaw[self.chirp_counter]
        )
        self.chirp_counter += 1
        self.pbar.update(1)  # Update progress bar

    def handle_land_state(self):
        """Handle the LAND state behavior"""
        if self.pbar is not None:  # Ensure progress bar is closed when landing
            self.pbar.close()
        self.land()
        if abs(self.vehicle_local_position.z) < self.landing_height_threshold:
            self.get_logger().info("Landing complete")
            self.disarm()
            rclpy.shutdown()

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()

        # State machine handling
        if self.current_state == DroneState.ARMING:
            self.handle_arming_state()
        elif self.current_state == DroneState.TAKEOFF:
            self.handle_takeoff_state()
        elif self.current_state == DroneState.CHIRP:
            self.handle_chirp_state()
        elif self.current_state == DroneState.LAND:
            self.handle_land_state()

import sys
def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
