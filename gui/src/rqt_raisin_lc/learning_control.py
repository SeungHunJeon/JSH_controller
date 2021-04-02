import os
import sys
import time

from ament_index_python import get_resource

import rclpy
from std_srvs.srv import Trigger
from rclpy.node import Node

from raisin_interfaces.srv import Vector3

from python_qt_binding import loadUi
from python_qt_binding.QtCore import Qt, QTimer, Slot, QProcess, QIODevice
from python_qt_binding.QtGui import QKeySequence
from python_qt_binding.QtWidgets import QShortcut, QWidget, QFileDialog
from rclpy.qos import QoSProfile
from rqt_gui_py.plugin import Plugin


class SetCommandAsync(Node):
    def __init__(self):
        super().__init__('set_command_async')
        self.cli = self.create_client(Vector3, 'raisin_learning_controller/set_command')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Set Command service not available, waiting again...')

    def send_request(self, command):
        self.future = self.cli.call_async(command)


class LearningControl(Plugin):

    def __init__(self, context):
        super(LearningControl, self).__init__(context)
        self.setObjectName('LearningControl')

        self._node = context.node

        self._widget = QWidget()
        _, package_path = get_resource('packages', 'rqt_raisin_lc')
        ui_file = os.path.join(
            package_path,
            'share',
            'rqt_raisin_lc',
            'resource',
            'LearningControl.ui')
        loadUi(ui_file, self._widget)
        self._widget.setObjectName('LearningControlUi')
        context.add_widget(self._widget)

        self._widget.set_command_push_button.pressed.connect(
            self._set_command_pressed)


    def _set_command_pressed(self, args=None):
        command = Vector3.Request()
        command.x = self._widget.x_lin_vel_box.value()
        command.y = self._widget.y_lin_vel_box.value()
        command.z = self._widget.z_ang_vel_box.value()

        self._node.get_logger().info(
            'x linear vel: %s, y linear vel: %s, z angular vel: %s.' %
            (command.x,
             command.y,
             command.z))

        can_cli = SetCommandAsync()
        can_cli.send_request(command)
        while rclpy.ok():
            rclpy.spin_once(can_cli)
            if can_cli.future.done():
                try:
                    response = can_cli.future.result()
                except Exception as e:
                    can_cli.get_logger().info('Service call failed %r' % (e,))
                else:
                    can_cli.get_logger().info(
                        'Result of set_command %s %s' %
                        (response.success, response.message))
                break
