#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# ----------------------------------------------------------------------------

dt-exec roslaunch --wait encoder_pose encoder_pose_node.launch veh:=$VEHICLE_NAME

# ----------------------------------------------------------------------------

# wait for app to end
dt-launchfile-join
