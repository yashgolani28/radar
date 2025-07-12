#!/bin/bash

exec ffmpeg \
-rtsp_transport tcp \
-i "rtsp://root:2024@192.168.1.241/axis-media/media.amp?streamprofile=stream1" \
-vf fps=5 \
-f image2 \
-update 1 \
-y /home/pi/radar/live.jpg
