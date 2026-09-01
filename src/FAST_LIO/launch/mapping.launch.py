import os.path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition

from launch_ros.actions import Node


def generate_launch_description():
    # 关键修正：包名如果是FAST_LIO（大写），这里要对应
    # 先确认包名：执行 ros2 pkg list | grep fast 看输出是 fast_lio 还是 FAST_LIO
    # 如果包名是小写fast_lio，保留下面这行；如果是大写FAST_LIO，改成 get_package_share_directory('FAST_LIO')
    package_path = get_package_share_directory('fast_lio')  # 包名通常是小写，路径是大写，这里区分开


    # 原有路径（不动）
    default_config_path = os.path.join(package_path, 'config')
    default_rviz_config_path = os.path.join(package_path, 'rviz', 'fastlio.rviz')

    # 原有参数
    use_sim_time = LaunchConfiguration('use_sim_time')
    config_path = LaunchConfiguration('config_path')
    config_file = LaunchConfiguration('config_file')
    rviz_use = LaunchConfiguration('rviz')
    rviz_cfg = LaunchConfiguration('rviz_cfg')
    
    # 栅格地图参数
    map_resolution = LaunchConfiguration('map_resolution')
    map_width = LaunchConfiguration('map_width')
    map_height = LaunchConfiguration('map_height')
    map_origin_x = LaunchConfiguration('map_origin_x')
    map_origin_y = LaunchConfiguration('map_origin_y')

    # 原有参数声明
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    declare_config_path_cmd = DeclareLaunchArgument(
        'config_path', default_value=default_config_path,
        description='Yaml config file path'
    )
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file', default_value='mid360.yaml',
        description='Config file'
    )
    declare_rviz_cmd = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Use RViz to monitor results'
    )
    declare_rviz_config_path_cmd = DeclareLaunchArgument(
        'rviz_cfg', default_value=default_rviz_config_path,
        description='RViz config file path'
    )
    
    # 栅格地图参数声明
    declare_map_resolution_cmd = DeclareLaunchArgument(
        'map_resolution', default_value='0.1',
        description='Grid map resolution (meters per cell)'
    )
    declare_map_width_cmd = DeclareLaunchArgument(
        'map_width', default_value='300',
        description='Grid map width (number of cells)'
    )
    declare_map_height_cmd = DeclareLaunchArgument(
        'map_height', default_value='300',
        description='Grid map height (number of cells)'
    )
    declare_map_origin_x_cmd = DeclareLaunchArgument(
        'map_origin_x', default_value='-15.0',
        description='Grid map origin X (meters)'
    )
    declare_map_origin_y_cmd = DeclareLaunchArgument(
        'map_origin_y', default_value='-15.0',
        description='Grid map origin Y (meters)'
    )

    # 原有fast_lio节点
    fast_lio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        parameters=[PathJoinSubstitution([config_path, config_file]),
                    {'use_sim_time': use_sim_time}],
        output='screen'
    )
   
    # 原有rviz节点
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(rviz_use)
    )

    ld = LaunchDescription()
    
    # 添加所有参数声明
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_config_path_cmd)
    ld.add_action(declare_config_file_cmd)
    ld.add_action(declare_rviz_cmd)
    ld.add_action(declare_rviz_config_path_cmd)
    ld.add_action(declare_map_resolution_cmd)
    ld.add_action(declare_map_width_cmd)
    ld.add_action(declare_map_height_cmd)
    ld.add_action(declare_map_origin_x_cmd)
    ld.add_action(declare_map_origin_y_cmd)

    # 添加所有节点
    ld.add_action(fast_lio_node)
    ld.add_action(rviz_node)

    return ld
