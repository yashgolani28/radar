#!/bin/bash

LOG_DIR="/home/pi/radar/logs"
DAYS_OLD=15

find "$LOG_DIR" -type f -mtime +$DAYS_OLD -exec rm -f {} \;
echo "$(date): Old logs cleaned (older than $DAYS_OLD days)" >> "$LOG_DIR/log_cleanup.log"
