#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# ----------------------------------------------------------------------------

dt-exec roslaunch --wait visual_slam visual_slam_node.launch veh:=$VEHICLE_NAME

# ----------------------------------------------------------------------------

# wait for app to end
dt-launchfile-join
