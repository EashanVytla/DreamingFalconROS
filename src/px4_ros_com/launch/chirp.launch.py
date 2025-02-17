import os
from launch import LaunchDescription, actions, substitutions
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, TimerAction
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    micro_ros_agent = ExecuteProcess(
        cmd=['xterm', '-e', 'MicroXRCEAgent', 'udp4', '-p', '8888', '-v'],
        shell=True,
        output='screen'
    )

    delay_timer = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='px4_ros_com',
                executable='chirp_test.py',
                output='screen',
                emulate_tty=True
            ),
        ]
    )

    delay_timer2 = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='px4_ros_com',
                executable='drf_agent.py',
                output='screen',
                emulate_tty=True
            )
        ]
    )

    return LaunchDescription([
        micro_ros_agent,
        delay_timer,
        delay_timer2
    ])