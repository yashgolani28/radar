#!/bin/bash

# Define variables
DATE=$(date +'%Y%m%d')
SRC="/home/pi/radar/radar_data.db"
DEST="gdrive:radar_backups/radar_backup_$DATE.db"

# Upload the backup
rclone copy "$SRC" "$DEST" --log-file=/home/pi/radar/backup_rclone.log --log-level INFO

# Optionally: clean up old backups on Drive (older than 7 days)
rclone delete --min-age 7d gdrive:radar_backups/
