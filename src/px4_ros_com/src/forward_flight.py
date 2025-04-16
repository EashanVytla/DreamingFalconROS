#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry, ActuatorOutputs
from message_filters import Subscriber, ApproximateTimeSynchronizer
import scipy
import numpy as np
from enum import Enum, auto
import math
from tqdm import tqdm
import yaml
from utils import AttrDict, l2_dist
from itertools import product
import time
import random
import torch
import pandas as pd
import utils
from scipy.spatial.transform import Rotation as R


class DroneState(Enum):
    ARMING = auto()
    TAKEOFF = auto()
    FORWARD_FLIGHT = auto()
    RESET = auto()
    LAND = auto()

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('offboard_control_forward_flight')

        with open("config.yaml", 'r') as file:
            config_dict = yaml.safe_load(file)

        self.config = AttrDict.from_dict(config_dict)

        # State management
        self.current_state = DroneState.ARMING
        self.target_takeoff_height = -10.0  # Target height for takeoff
        self.takeoff_height_threshold = -9.8  # Height at which takeoff is considered complete
        self.landing_height_threshold = -0.2  # Height at which landing is considered complete

        self.origin = np.array([0.0, 0.0, self.target_takeoff_height], dtype=np.float32)
        self.prod_cnt = 0

        self.steady_velo = 2.0

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
        
        self.odometry_subscriber = Subscriber(
            self,
            VehicleOdometry,
            'fmu/out/vehicle_odometry',
            qos_profile=qos_profile,
        )

        self.actuator_subscriber = Subscriber(
            self,
            ActuatorOutputs,
            'fmu/out/actuator_outputs',
            qos_profile=qos_profile,
        )

        self.time_sync = ApproximateTimeSynchronizer(
            [self.odometry_subscriber, self.actuator_subscriber],
            queue_size=5,
            slop=0.05,
            allow_headerless=True
        )

        self.time_sync.registerCallback(self.data_callback)

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
        
        self.start_time = time.time()

        # Create a timer to publish control commands
        self.timer = self.create_timer(self.config.physics.refresh_rate, self.timer_callback)
        self.last_timestamp = 0.0
        self.device = self.config.device
        self.start_time = time.time()
        self.start_act = False
        self.state_dim = self.config.force_model.state_dim
        self.action_dim = self.config.force_model.action_dim
        self.device = self.config.device

        '''
        State:
            1) Position (NED) 0-3
            2) Velocity (body) 12-15
            5) Attitude (Euler) 6-9
            6) Angular Velocity (body) 9-12
        '''

        self.state = torch.zeros((12), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((self.action_dim), dtype=torch.float32, device=self.device)



    def data_callback(self, odo_msg, act_msg):
        current_timestamp = odo_msg.timestamp
        dt = 0.0
        dt = (current_timestamp - self.last_timestamp) / 1e6
        if (dt < self.config.physics.refresh_rate - (0.05 * self.config.physics.refresh_rate)) or self.current_state != DroneState.FORWARD_FLIGHT:
            return

        print("logging")
        self.state[0:3] = torch.tensor(odo_msg.position, dtype=torch.float32, device=self.device)

        self.state[3:6] = torch.matmul(
            torch.tensor(R.from_quat(odo_msg.q, scalar_first=True).as_matrix().T, dtype=torch.float32, device=self.device),
            torch.tensor(odo_msg.velocity, dtype=torch.float32, device=self.device)
        )

        self.state[6:9] = utils.quat_to_euler(odo_msg.q, device=self.device)

        if not hasattr(self, 'prev_yaw'):
            self.prev_yaw = self.state[8].clone()
        else:
            self.state[8] = utils.unwrap_angle(self.state[8], self.prev_yaw)
            self.prev_yaw = self.state[8].clone()

        self.state[9:12] = torch.tensor(odo_msg.angular_velocity, dtype=torch.float32, device=self.device)

        self.action = torch.tensor(act_msg.output[:4], dtype=torch.float32, device=self.device)

        # Log data to CSV file
        if not hasattr(self, 'csv_file'):
            import csv
            import os
            log_dir = os.path.join(os.getcwd(), "flight_logs")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.csv_file = open(f"{log_dir}/flight_data_{timestamp}.csv", 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            # Write header
            self.csv_writer.writerow([
                'timestamp', 'dt',
                'x', 'y', 'z',  # Position
                'vx', 'vy', 'vz',  # Velocity
                'roll', 'pitch', 'yaw',  # Attitude
                'roll_rate', 'pitch_rate', 'yaw_rate',  # Angular rates
                'motor1', 'motor2', 'motor3', 'motor4'  # Motor outputs
            ])
        
        # Write data row
        self.csv_writer.writerow([
            current_timestamp, dt,
            *self.state.cpu().numpy().tolist(),  # Unpack all state values
            *self.action.cpu().numpy().tolist()  # Unpack all action values
        ])
        
        # Ensure data is written to disk periodically
        if self.publish_counter % 100 == 0:
            self.csv_file.flush()
        
        self.last_timestamp = current_timestamp

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
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.direct_actuator = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_position_setpoint(self, x: float, y: float, z: float, yaw=1.57):
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()
        msg.position = [x, y, z]
        msg.velocity = [0.0, 0.0, 0.0]
        msg.acceleration = [0.0, 0.0, 0.0]
        msg.yaw = yaw  # (90 degree)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z]}", throttle_duration_sec=1)

    def publish_rate_setpoint(self, ax: float, ay: float, az: float, yaw: float, vx=0, vy=0, vz=0):
        msg = TrajectorySetpoint()
        msg.velocity = [vx, vy, vz]
        msg.acceleration = [ax, ay, az]
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_velo_setpoint(self, vx=0.0, vy=0.0, vz=0.0):
        msg = TrajectorySetpoint()
        msg.position = [math.nan, math.nan, math.nan]
        msg.velocity = [vx, vy, vz]
        msg.yaw = 0.0
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
        dist = abs(self.vehicle_local_position.z - self.target_takeoff_height)
        print(f"Dist to target altitude: {dist}")
        
        if dist < 1.0:
            self.get_logger().info("Takeoff complete, starting forward flight")
            self.current_state = DroneState.FORWARD_FLIGHT
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)

    def handle_ff_state(self, vx: float, vy: float, vz: float):
        """Handle the FORWARD_FLIGHT state behavior"""
        self.publish_velo_setpoint(
            vx = vx,
            vy = vy,
            vz = vz
        )

    def handle_land_state(self):
        """Handle the LAND state behavior"""
        print("LANDING!!!")
        self.land()
        if abs(self.vehicle_local_position.z) < self.landing_height_threshold:
            self.get_logger().info("Landing complete")
            self.disarm()
            rclpy.shutdown()

    def handle_reset_state(self):
        pos = np.array([self.vehicle_local_position.x, self.vehicle_local_position.y, self.vehicle_local_position.z], dtype=np.float32)
        
        dist = l2_dist(pos, self.origin)

        if dist > 2.0:
            self.publish_position_setpoint(self.origin[0].item(), self.origin[1].item(), self.origin[2].item(), math.radians(90))
        else:
            self.current_state = self.cache_state

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()

        # State machine handling
        if self.current_state == DroneState.ARMING:
            self.handle_arming_state()
        elif self.current_state == DroneState.TAKEOFF:
            self.handle_takeoff_state()
        elif self.current_state == DroneState.FORWARD_FLIGHT:
            self.handle_ff_state(vx=self.steady_velo, vy=0.0, vz=0.0)
        elif self.current_state == DroneState.RESET:
            self.handle_reset_state()
        elif self.current_state == DroneState.LAND:
            self.handle_land_state()

import sys
def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    timeout = 3.2

    while rclpy.ok():
        if offboard_control.current_state != DroneState.FORWARD_FLIGHT:
            start_time = time.time()
        rclpy.spin_once(offboard_control)

        if time.time() - start_time > timeout:
            print("\nTimeout reached! Shutting down forward flight.")
            break

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
