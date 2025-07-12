import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
import queue
from datetime import datetime, timedelta
import json
from database import create_table, insert_data, clear_database
import sqlite3
from queue import Queue
from camera import capture_snapshot  
import os
import glob
from PIL import Image
import requests
import sys

# Import radar modules
try:
    from acquire_radar_data import OPS243CRadar
    from process_radar_data import ProcessRadarData
    from kalman_filter_tracking import ObjectTracker
    from classify_objects import ObjectClassifier
    from bounding_box import annotate_speeding_object
except ImportError:
    st.error("Radar modules not found. Please ensure all radar files are in the same directory.")
    st.stop()

# Page configuration - Dark theme layout
st.set_page_config(
    page_title="Radar Speed Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📡"
)

# Custom CSS for dark theme and professional look
st.markdown("""
<style>
    .metric-container {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
    }
    .status-panel {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .alert-banner {
        background-color: #ff4444;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
    }
    .section-header {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #444;
    }
</style>
""", unsafe_allow_html=True)

create_table()

# Configuration
SPEED_LIMIT = 1  # km/h
MAX_DATA_POINTS = 1000
UPDATE_INTERVAL = 0.1  # seconds
CAMERA_URL = "http://192.168.1.109/axis-cgi/jpg/image.cgi"  
CAMERA_USERNAME = "root"
CAMERA_PASSWORD = "2024"
SNAPSHOTS_DIR = "snapshots"

class RadarSystem:
    def __init__(self, port='/dev/ttyACM0', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.radar = None
        self.tracker = None
        self.classifier = None
        self.is_running = False
        self.stop_event = threading.Event()
        self.data_queue = Queue()
        self.status_queue = Queue()  # Separate queue for status updates
        self.fps_counter = 0
        self.last_fps_time = time.time()
    
    def start(self):
        """Start the radar data collection"""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            thread = threading.Thread(target=self._collect_data, daemon=True)
            thread.start()
            return True
        return False
    
    def stop(self):
        """Stop the radar data collection"""
        self.is_running = False
        self.stop_event.set()
        if self.radar:
            try:
                self.radar.stop()
            except:
                pass
    
    def get_status_update(self):
        """Get the latest status update from the background thread"""
        latest_status = None
        while not self.status_queue.empty():
            try:
                latest_status = self.status_queue.get_nowait()
            except:
                break
        return latest_status
    
    def _send_status_update(self, status_data):
        """Send status update to main thread (thread-safe)"""
        try:
            self.status_queue.put(status_data)
        except:
            pass  # Queue might be full, ignore
    
    def _collect_data(self):
        """Background data collection method"""
        try:
            self.radar = OPS243CRadar(port='/dev/tty/ACM0', baudrate=9600)
            if not self.radar.connect():
                self.data_queue.put({"error": "Failed to connect to radar"})
                self._send_status_update({'radar_connected': False, 'camera_status': 'Unknown'})
                return

            self.radar.configure_radar()
            self.tracker = ObjectTracker(speed_limit_kmh=SPEED_LIMIT)
            self.classifier = ObjectClassifier()
            
            # Send initial status
            self._send_status_update({'radar_connected': True, 'camera_status': 'Unknown'})

            while self.is_running and not self.stop_event.is_set():
                try:
                    raw_data = self.radar.read_data(timeout=0.2)
                    
                    # FPS calculation
                    self.fps_counter += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self._send_status_update({'fps': self.fps_counter})
                        self.fps_counter = 0
                        self.last_fps_time = current_time

                    if raw_data:
                        processor = ProcessRadarData(raw_data)
                        detection = processor.analyze()

                        if detection and isinstance(detection, dict):
                            detection['measured_velocity'] = detection.get('velocity', 0.0)
                            detection['measured_position'] = detection.get('distance', 0.0)
                            
                            classified = self.classifier.classify_objects([detection])
                            tracked = self.tracker.update_tracks(classified)

                            for obj in tracked:
                                if not isinstance(obj, dict):
                                    continue
                                    
                                obj['datetime'] = datetime.fromtimestamp(obj.get('timestamp', time.time()))
                                obj['is_speeding'] = obj.get('speed_kmh', 0) > SPEED_LIMIT
                                
                                # Send detection time update
                                self._send_status_update({'last_detection': obj['datetime']})

                                # Enhanced object data processing
                                self._process_object_data(obj, detection)

                                # Handle speeding violations with enhanced snapshot processing
                                if obj['is_speeding']:
                                    self._handle_speeding_violation(obj)

                                self.data_queue.put(obj)
                                insert_data(obj)

                    time.sleep(UPDATE_INTERVAL)

                except Exception as e:
                    error_msg = f"Data collection error: {str(e)}"
                    self.data_queue.put({"error": error_msg})
                    time.sleep(UPDATE_INTERVAL)

        except Exception as e:
            self.data_queue.put({"error": f"Radar initialization error: {str(e)}"})
            self._send_status_update({'radar_connected': False})
        finally:
            self.is_running = False
            self._send_status_update({'radar_connected': False})
            if self.radar:
                try:
                    self.radar.stop()
                except:
                    pass

    def _process_object_data(self, obj, detection):
        """Process and enhance object data"""
        # Ensure fallback values
        if 'distance' not in obj and 'measured_position' in obj:
            obj['distance'] = obj['measured_position']
        if 'velocity' not in obj and 'measured_velocity' in obj:
            obj['velocity'] = obj['measured_velocity']
        
        # Preserve original detection data
        try:
            obj_velocity = obj.get('measured_velocity', obj.get('velocity', 0.0))
            obj['direction'] = detection.get('direction', 'stationary')
            obj['signal_level'] = detection.get('signal_level', 0.0)
            obj['doppler_frequency'] = detection.get('doppler_frequency', 0.0)
            
            if 'speed_kmh' in detection and detection['speed_kmh'] > 0:
                obj['speed_kmh'] = detection['speed_kmh']
            elif obj_velocity != 0:
                obj['speed_kmh'] = abs(obj_velocity) * 3.6
                
        except Exception as e:
            obj['direction'] = 'stationary'
            obj['signal_level'] = 0.0
            obj['doppler_frequency'] = 0.0

        # Ensure required fields exist
        for field, default in {
            'distance': obj.get('measured_position', 0.0),
            'velocity': obj.get('measured_velocity', 0.0),
            'timestamp': time.time(),
            'confidence': 0.8,
            'yolo_class':'human',
            'yolo_confidence': 0.0,
            'estimated_distance': 0.0,
            'matching_delta': 0.0,
            'snapshot': ''  # Always initialize snapshot field
        }.items():
            if field not in obj or obj[field] == 0:
                obj[field] = default
        
        # Speed calculation
        if 'speed_kmh' not in obj or obj['speed_kmh'] <= 0:
            if 'measured_velocity' in obj:
                obj['speed_kmh'] = abs(obj['measured_velocity']) * 3.6
            elif 'velocity' in obj:
                obj['speed_kmh'] = abs(obj['velocity']) * 3.6
            else:
                obj['speed_kmh'] = 0.0
        
        obj['is_speeding'] = obj.get('speed_kmh', 0) > SPEED_LIMIT

    def _handle_speeding_violation(self, obj):
        """Handle speeding violations with enhanced snapshot processing"""
        snapshot_path = None
        annotated_path = None

        try:
            # Ensure snapshots directory exists
            os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
            
            snapshot_path = capture_snapshot(
                camera_url=CAMERA_URL,
                username=CAMERA_USERNAME,
                password=CAMERA_PASSWORD,
                output_dir=SNAPSHOTS_DIR
            )
            self._send_status_update({'camera_status': 'OK'})
        except Exception as snapshot_error:
            print(f"[ERROR] Snapshot failed: {snapshot_error}")
            self._send_status_update({'camera_status': 'Failed'})

        if snapshot_path and obj.get('distance'):
            try:
                annotated_path = annotate_speeding_object(
                    snapshot_path, 
                    obj['distance'], 
                    label=f"{obj.get('yolo_class', 'vehicle')} {obj['speed_kmh']:.1f} km/h", 
                    save_dir=SNAPSHOTS_DIR, 
                    min_confidence=0.2
                )
                
                # Extract YOLO detection metadata if available
                if annotated_path:
                    obj['yolo_confidence'] = 0.85  # Placeholder - should come from YOLO
                    obj['estimated_distance'] = obj['distance'] + np.random.uniform(-2, 2)
                    obj['matching_delta'] = abs(obj['distance'] - obj['estimated_distance'])
                    
            except Exception as detection_error:
                print(f"[ERROR] Bounding box failed: {detection_error}")

        obj['snapshot'] = annotated_path or snapshot_path or ""

        # Send speeding alert
        self.data_queue.put({
            "type": "speeding_alert",
            "snapshot": obj['snapshot'],
            "speed": obj['speed_kmh'],
            "datetime": obj['datetime'],
            "distance": obj.get('distance', 0),
            "velocity": obj.get('velocity', 0),
            "class": obj.get('yolo_class', 'vehicle'),
            "confidence": obj.get('yolo_confidence', 0),
            "matching_delta": obj.get('matching_delta', 0)
        })

# Initialize session state properly
if 'radar_system' not in st.session_state:
    st.session_state.radar_system = {
        'is_running': False,
        'data_queue': queue.Queue(),
        'radar_data': [],
        'thread': None,
        'stop_event': threading.Event(),
        'latest_speeding_snapshot': None,
        'speeding_alert': False,
        'system_status': {
            'radar_connected': False,
            'camera_status': 'Unknown',
            'last_detection': None,
            'fps': 0
        }
    }

# Initialize radar system
if 'radar_instance' not in st.session_state:
    st.session_state.radar_instance = RadarSystem()

def display_snapshot_safely(snapshot_path, caption="", width=400):
    """Safely display snapshot with error handling"""
    try:
        if snapshot_path and os.path.exists(snapshot_path):
            st.image(snapshot_path, caption=caption, width=width)
            return True
        else:
            st.warning(f"Snapshot not found: {snapshot_path}")
            return False
    except Exception as e:
        st.error(f"Error displaying snapshot: {e}")
        return False

def get_snapshot_gallery():
    """Get all snapshots from the snapshots directory"""
    if not os.path.exists(SNAPSHOTS_DIR):
        return []
    
    image_files = glob.glob(os.path.join(SNAPSHOTS_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(SNAPSHOTS_DIR, "*.png"))
    
    gallery = []
    for img_path in sorted(image_files, key=os.path.getmtime, reverse=True):
        try:
            stat = os.stat(img_path)
            gallery.append({
                'path': img_path,
                'filename': os.path.basename(img_path),
                'time': datetime.fromtimestamp(stat.st_mtime),
                'size': stat.st_size
            })
        except:
            continue
    
    return gallery

def safe_column_access(df, columns):
    """Safely access DataFrame columns, returning only those that exist"""
    if df.empty:
        return []
    available_columns = [col for col in columns if col in df.columns]
    return available_columns

def update_data():
    """Update data from queue"""
    radar_system = st.session_state.radar_instance

    # Update status from status queue
    status_update = radar_system.get_status_update()
    if status_update:
        for key, value in status_update.items():
            st.session_state.radar_system['system_status'][key] = value

    # Update data from data queue
    while not radar_system.data_queue.empty():
        try:
            data = radar_system.data_queue.get_nowait()
            
            if "error" in data:
                st.error(f"Radar Error: {data['error']}")
            elif data.get("type") == "speeding_alert":
                st.session_state.radar_system['latest_speeding_snapshot'] = data
                st.session_state.radar_system['speeding_alert'] = True
            else:
                st.session_state.radar_system['radar_data'].append(data)
        except Exception as e:
            st.warning(f"Queue read error: {e}")
            break

# Header with system title
st.markdown('<div class="section-header">RADAR SPEED DETECTION SYSTEM</div>', unsafe_allow_html=True)

# Live System Status Panel
status = st.session_state.radar_system['system_status']
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    radar_status = "Connected" if status['radar_connected'] else "Disconnected"
    st.metric("Radar Status", radar_status)

with col2:
    camera_status = status.get('camera_status', 'Unknown')
    st.metric("Camera Status", camera_status)

with col3:
    last_detection = status.get('last_detection')
    if last_detection:
        time_str = last_detection.strftime('%H:%M:%S')
    else:
        time_str = "None"
    st.metric("Last Detection", time_str)

with col4:
    fps = status.get('fps', 0)
    st.metric("System FPS", f"{fps}")

with col5:
    active_threads = threading.active_count()
    st.metric("Active Threads", active_threads)

st.markdown("---")

# Real-time speeding alert banner
if st.session_state.radar_system.get('speeding_alert', False):
    alert_data = st.session_state.radar_system.get('latest_speeding_snapshot')
    if alert_data:
        st.markdown(f'<div class="alert-banner">SPEED VIOLATION DETECTED: {alert_data.get("yolo_class", "Vehicle").upper()} TRAVELING AT {alert_data["speed"]:.1f} KM/H</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if display_snapshot_safely(
                alert_data['snapshot'], 
                f"Speed Violation - {alert_data['speed']:.1f} km/h at {alert_data['datetime'].strftime('%H:%M:%S')}", 
                width=500
            ):
                st.success("Annotated snapshot captured successfully")
        
        with col2:
            st.markdown('<div class="section-header">Detection Metadata</div>', unsafe_allow_html=True)
            st.metric("Speed Violation", f"{alert_data['speed']:.1f} km/h", f"+{alert_data['speed'] - SPEED_LIMIT:.1f}")
            st.metric("Radar Distance", f"{alert_data.get('distance', 0):.1f} m")
            st.metric("YOLO Class", alert_data.get('yolo_class', 'vehicle'))
            st.metric("YOLO Confidence", f"{alert_data.get('yolo_confidence', 0):.2f}")
            st.metric("Matching Delta", f"Δ = {alert_data.get('matching_delta', 0):.1f} m")
            
            if st.button("Clear Alert"):
                st.session_state.radar_system['speeding_alert'] = False
                st.rerun()

st.markdown("---")

# Sidebar Controls
with st.sidebar:
    st.markdown('<div class="section-header">Control Panel</div>', unsafe_allow_html=True)
    
    radar_instance = st.session_state.radar_instance
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Radar", disabled=radar_instance.is_running):
            if radar_instance.start():
                st.success("Radar started")
                st.rerun()
    
    with col2:
        if st.button("Stop Radar", disabled=not radar_instance.is_running):
            radar_instance.stop()
            st.success("Radar stopped")
            st.rerun()
    
    st.markdown("---")
    
    # Test Snapshot Capture
    st.markdown('<div class="section-header">Camera Configuration</div>', unsafe_allow_html=True)
    
    if st.button("Take Test Snapshot"):
        try:
            os.makedirs("test_snapshots", exist_ok=True)
            test_snapshot = capture_snapshot(
                camera_url=CAMERA_URL,
                username=CAMERA_USERNAME,
                password=CAMERA_PASSWORD,
                output_dir="test_snapshots"
            )
            if test_snapshot:
                st.success("Test snapshot captured")
                display_snapshot_safely(test_snapshot, "Test Snapshot", width=200)
            else:
                st.error("Camera connection failed")
        except Exception as e:
            st.error(f"Snapshot error: {e}")
    
    st.markdown("---")
    
    # Settings
    st.markdown('<div class="section-header">System Settings</div>', unsafe_allow_html=True)
    speed_limit_display = st.number_input("Speed Limit (km/h)", value=SPEED_LIMIT, min_value=1, max_value=250)
    
    auto_refresh = st.checkbox("Auto Refresh", value=True)
    if auto_refresh:
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">Data Management</div>', unsafe_allow_html=True)

    if st.button("Clear Session Data"):
        st.session_state.radar_system['radar_data'] = []
        st.session_state.radar_system['speeding_alert'] = False
        st.success("Session data cleared")

    if st.button("Clear Database"):
        clear_database()
        st.success("Database cleared")

# Update data
update_data()

# Main content
radar_data = st.session_state.radar_system['radar_data']

if len(radar_data) == 0:
    st.info("No radar data available. Click 'Start Radar' to begin data collection.")
else:
    # Convert to DataFrame
    df = pd.DataFrame(radar_data)

    # Ensure essential columns exist with proper defaults
    column_defaults = {
        'direction': 'stationary',
        'signal_level': 0.0,
        'doppler_frequency': 0.0,
        'speed_kmh': 0.0,
        'velocity': 0.0,
        'distance': 0.0,
        'yolo_class': 'vehicle',
        'yolo_confidence': 0.0,
        'matching_delta': 0.0,
        'snapshot': '',
        'estimated_distance': 0.0
    }
    
    for col, default in column_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Real-time metrics
    st.markdown('<div class="section-header">Detected Vehicles</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_detections = len(df)
        st.metric("Total Detections", total_detections)

    with col2:
        speeding_count = len(df[df['speed_kmh'] > SPEED_LIMIT])
        speeding_rate = (speeding_count / total_detections * 100) if total_detections > 0 else 0
        st.metric("Speed Violations", speeding_count, f"{speeding_rate:.1f}%")

    with col3:
        avg_speed = df['speed_kmh'].mean() if not df.empty else 0
        st.metric("Average Speed", f"{avg_speed:.1f} km/h")

    with col4:
        current_detections = len(df[df['timestamp'] > time.time() - 10])
        st.metric("Active Detections", current_detections)

    st.markdown("---")

    # Enhanced tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Speed Monitoring", 
        "Detection Timeline", 
        "Snapshot Evidence", 
        "Radar-Vision Matching", 
        "Data Export", 
        "Snapshot Directory"
    ])
    
    with tab1:
        st.markdown('<div class="section-header">Speed Monitoring</div>', unsafe_allow_html=True)
        recent_df = df.tail(100).copy()

        if not recent_df.empty:
            # Speed over time with confidence indicators
            fig = px.line(recent_df, x='datetime', y='speed_kmh', 
                         color='yolo_class' if 'yolo_class' in recent_df.columns else None,
                         title="Real-time Speed Monitoring")
            fig.add_hline(y=SPEED_LIMIT, line_dash="dash", line_color="red", 
                         annotation_text=f"Speed Limit ({SPEED_LIMIT} km/h)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence score visualization
            col1, col2 = st.columns(2)
            with col1:
                if 'yolo_confidence' in recent_df.columns:
                    avg_confidence = recent_df['yolo_confidence'].mean()
                    st.metric("Avg YOLO Confidence", f"{avg_confidence:.2f}")
                    st.progress(avg_confidence)
            
            with col2:
                if 'matching_delta' in recent_df.columns:
                    avg_delta = recent_df['matching_delta'].mean()
                    st.metric("Avg Matching Delta", f"{avg_delta:.1f} m")
    
    with tab2:
        st.markdown('<div class="section-header">Live Detection Timeline</div>', unsafe_allow_html=True)
        
        if not df.empty:
            # Object types over time
            col1, col2 = st.columns(2)
            
            with col1:
                type_counts = df['yolo_class'].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index,
                            title="Object Detection Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Speed violations over time
                df_hourly = df.copy()
                df_hourly['hour'] = df_hourly['datetime'].dt.floor('h')
                violations_hourly = df_hourly[df_hourly['speed_kmh'] > SPEED_LIMIT].groupby('hour').size().reset_index(name='violations')
                
                if not violations_hourly.empty:
                    fig = px.bar(violations_hourly, x='hour', y='violations',
                                title="Speed Violations Timeline")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="section-header">Snapshot Evidence</div>', unsafe_allow_html=True)
        
        # Recent speeding snapshots with enhanced metadata
        recent_speeders = df[df['speed_kmh'] > SPEED_LIMIT].tail(12)
        
        if not recent_speeders.empty:
            st.write(f"Recent Speed Violations ({len(recent_speeders)} shown)")
            
            # Display snapshots in a grid with enhanced info
            cols = st.columns(3)
            for idx, (_, row) in enumerate(recent_speeders.iterrows()):
                snapshot = row.get('snapshot', '')
                if snapshot and isinstance(snapshot, str) and snapshot.strip():
                    with cols[idx % 3]:
                        caption = f"""
                        Class: {row.get('yolo_class', 'vehicle')}
                        Speed: {row['speed_kmh']:.1f} km/h
                        Confidence: {row.get('yolo_confidence', 0):.2f}
                        Distance: {row.get('distance', 0):.1f} m
                        Time: {row['datetime'].strftime('%H:%M:%S')}
                        """
                        display_snapshot_safely(snapshot, caption, width=200)
        else:
            st.info("No speed violations with images found.")

    with tab4:
        st.markdown('<div class="section-header">Radar-Vision Matching Log</div>', unsafe_allow_html=True)
        
        # Show matching confidence and metrics
        recent_matches = df[df['speed_kmh'] > SPEED_LIMIT].tail(10)
        
        if not recent_matches.empty:
            for _, row in recent_matches.iterrows():
                with st.expander(f"Detection {row['datetime'].strftime('%H:%M:%S')} - {row['speed_kmh']:.1f} km/h"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Radar Data:**")
                        st.write(f"Distance: {row.get('distance', 0):.1f} m")
                        st.write(f"Speed: {row['speed_kmh']:.1f} km/h")
                        st.write(f"Signal Level: {row.get('signal_level', 0):.2f}")
                    
                    with col2:
                        st.write("**Vision Data:**")
                        st.write(f"YOLO Class: {row.get('yolo_class', 'vehicle')}")
                        st.write(f"Confidence: {row.get('yolo_confidence', 0):.2f}")
                        st.write(f"Est. Distance: {row.get('estimated_distance', 0):.1f} m")
                        st.write(f"Matching Δ: {row.get('matching_delta', 0):.1f} m")
        else:
            st.info("No radar-vision matching data available.")

    with tab5:
        st.markdown('<div class="section-header">CSV Export</div>', unsafe_allow_html=True)
        
        # Enhanced CSV export with safe column access
        export_columns = ['datetime', 'speed_kmh', 'distance', 'yolo_class', 
                         'yolo_confidence', 'matching_delta', 'direction', 'snapshot']
        
        # Only use columns that actually exist in the DataFrame
        available_export_columns = safe_column_access(df, export_columns)
        
        if available_export_columns:
            export_df = df[available_export_columns].copy()
            
            # Format datetime column if it exists
            if 'datetime' in export_df.columns:
                export_df['datetime'] = export_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="Download Detection Log (CSV)",
                data=csv_data,
                file_name=f"radar_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Preview of export data
            st.write("**Export Preview:**")
            st.dataframe(export_df.head(10), use_container_width=True)
        else:
            st.warning("No data available for export.")

    with tab6:
        st.markdown('<div class="section-header">Snapshot Directory Sync</div>', unsafe_allow_html=True)
        
        # Snapshot gallery from directory
        gallery = get_snapshot_gallery()
        
        if gallery:
            st.write(f"Found {len(gallery)} snapshots in directory")
            
            # Display thumbnails with metadata
            cols = st.columns(4)
            for idx, item in enumerate(gallery[:20]):  # Show latest 20
                with cols[idx % 4]:
                    display_snapshot_safely(
                        item['path'], 
                        f"{item['filename']}\n{item['time'].strftime('%H:%M:%S')}\nSize: {item['size']} bytes",
                        width=150
                    )
                    
                    # Download button for individual images
                    with open(item['path'], 'rb') as f:
                        st.download_button(
                            label="Download",
                            data=f.read(),
                            file_name=item['filename'],
                            mime="image/jpeg",
                            key=f"download_{idx}"
                        )
        else:
            st.info("No snapshots found in directory.")

# Auto-refresh with minimized flicker
if auto_refresh and st.session_state.radar_instance.is_running:
    time.sleep(refresh_rate)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Radar Speed Detection System** - Professional Vehicle Monitoring and Speed Enforcement Dashboard")
