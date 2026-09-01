import os
import platform
import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Setup library path
arch = platform.machine().replace('amd64', 'x86_64').replace('arm64', 'aarch64')
lib_path = os.path.abspath(
    f'{os.path.dirname(__file__)}/../../../../lib/zsl-1/{arch}')
sys.path.insert(0, lib_path)

import mc_sdk_zsl_1_py

class AgibotCommandListener(Node):
    def __init__(self, app):
        super().__init__('agibot_command_listener')
        self.app = app
        self.subscription = self.create_subscription(
            String,
            '/agibot_cmd',
            self.command_callback,
            10
        )
        self.get_logger().info("Started listening to /agibot_cmd topic")

    def command_callback(self, msg):
        """Process received command from ROS2 topic."""
        choice = msg.data.strip()
        self.get_logger().info(f"Received command: {choice}")

        try:
            if choice == '1':
                print("Executing: Stand Up")
                self.app.standUp()
                time.sleep(3)
            elif choice == '2':
                print("Executing: Lie Down")
                self.app.lieDown()
                time.sleep(3)
            elif choice == '3':
                print("Executing: Move Forward (for 2s)")
                self.app.move(0.2, 0, 0)
                time.sleep(2)
                self.app.move(0, 0, 0)  # Stop
            elif choice == '4':
                print("Executing: Move Backward (for 2s)")
                self.app.move(-0.2, 0, 0)
                time.sleep(2)
                self.app.move(0, 0, 0)  # Stop
            elif choice == '5':
                print("Executing: Move Left (for 2s)")
                self.app.move(0, 0.2, 0)
                time.sleep(2)
                self.app.move(0, 0, 0)  # Stop
            elif choice == '6':
                print("Executing: Move Right (for 2s)")
                self.app.move(0, -0.2, 0)
                time.sleep(2)
                self.app.move(0, 0, 0)  # Stop
            elif choice == '7':
                print("Executing: Turn Left (for 2s)")
                self.app.move(0, 0, 0.3)
                time.sleep(2)
                self.app.move(0, 0, 0)  # Stop
            elif choice == '8':
                print("Executing: Turn Right (for 2s)")
                self.app.move(0, 0, -0.3)
                time.sleep(2)
                self.app.move(0, 0, 0)  # Stop
            elif choice == '9':
                print("Executing: Jump")
                self.app.jump()
                time.sleep(4)
            elif choice == '10':
                print("Executing: Front Jump")
                self.app.frontJump()
                time.sleep(4)
            elif choice == '11':
                print("Executing: Backflip")
                self.app.backflip()
                time.sleep(4)
            elif choice == '12':
                print("Executing: Shake Hand")
                self.app.shakeHand()
                time.sleep(4)
            elif choice == '13':
                print("Executing: Attitude Control (for 4s)")
                self.app.attitudeControl(0.1, 0.1, 0.1, 0.1)
                time.sleep(4)
                self.app.standUp()  # Return to a stable state
                time.sleep(2)
            elif choice == '0':
                print("Received exit command. Robot will lie down.")
                self.app.lieDown()
                time.sleep(3)
                rclpy.shutdown()
            else:
                # Try to parse continuous movement commands "vx vy vyaw"
                try:
                    values = list(map(float, choice.split()))
                    if len(values) == 3:
                        vx, vy, vyaw = values
                        print(f"Executing continuous move: vx={vx}, vy={vy}, vyaw={vyaw}")
                        self.app.move(vx, vy, vyaw)
                    else:
                        print(f"Invalid command format: {choice}")
                except ValueError:
                    print(f"Invalid choice: {choice}. Please send 0-13 or 'vx vy vyaw' for continuous movement.")
                    return

            # Ensure robot is in a stable standing state after most actions
            if choice not in ['1', '2', '0'] and len(choice.strip().split()) != 3:
                self.app.standUp()
                time.sleep(2)

        except Exception as e:
            self.get_logger().error(f"Error executing command: {e}")


def main():
    """Main function to run the ROS2 command listener demo."""
    try:
        # Initialize ROS2
        rclpy.init(args=None)

        app = mc_sdk_zsl_1_py.HighLevel()
        # Use 127.0.0.1 for both local and robot IP for simulation/local testing
        app.initRobot("192.168.234.18", 43988, "192.168.234.1")
        print("Successfully initialized robot connection.")
        print(lib_path)

        # Create command listener node
        listener = AgibotCommandListener(app)

        # Spin ROS2 node to listen for commands
        rclpy.spin(listener)

        # Cleanup
        listener.destroy_node()
        rclpy.shutdown()
        print("Shutting down...")

    except ImportError as e:
        print(f"ImportError: {e}")
        print("Error: The 'mc_sdk_zsl_1_py' module could not be found or ROS2 is not available.")
        print(
            f"Please ensure the library for your architecture ('{arch}') is in the path: {lib_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
