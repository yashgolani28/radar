import os
import requests
import sqlite3
from datetime import datetime
import time

def check_snapshot_issue():
    """Debug snapshot capture issues"""
    print("🔍 SNAPSHOT DEBUG ANALYSIS")
    print("=" * 50)
    
    # 1. Check if snapshots directory exists
    print("\n1. Checking snapshots directory...")
    snapshots_dir = "snapshots"
    if os.path.exists(snapshots_dir):
        files = os.listdir(snapshots_dir)
        print(f"   ✅ Directory exists: {snapshots_dir}")
        print(f"   📁 Files in directory: {len(files)}")
        if files:
            print(f"   📄 Files: {files[:5]}...")  # Show first 5 files
        else:
            print("   ⚠️  Directory is empty")
    else:
        print(f"   ❌ Directory does not exist: {snapshots_dir}")
        try:
            os.makedirs(snapshots_dir)
            print(f"   ✅ Created directory: {snapshots_dir}")
        except Exception as e:
            print(f"   ❌ Failed to create directory: {e}")
    
    # 2. Check database for speeding violations
    print("\n2. Checking database for speeding violations...")
    try:
        conn = sqlite3.connect("radar_data.db")
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='radar_data'")
        if cursor.fetchone():
            print("   ✅ Database table exists")
            
            # Check for speeding violations
            cursor.execute("SELECT COUNT(*) FROM radar_data WHERE speed_kmh > 1")
            speeding_count = cursor.fetchone()[0]
            print(f"   📊 Speeding violations in DB: {speeding_count}")
            
            # Check for snapshots in database
            cursor.execute("SELECT COUNT(*) FROM radar_data WHERE snapshot IS NOT NULL AND snapshot != ''")
            snapshots_in_db = cursor.fetchone()[0]
            print(f"   📷 Records with snapshots in DB: {snapshots_in_db}")
            
            # Show recent speeding violations
            cursor.execute("SELECT timestamp, speed_kmh, snapshot FROM radar_data WHERE speed_kmh > 1 ORDER BY timestamp DESC LIMIT 5")
            recent_violations = cursor.fetchall()
            print(f"   📋 Recent violations:")
            for i, (ts, speed, snapshot) in enumerate(recent_violations):
                snapshot_status = "✅ Has snapshot" if snapshot else "❌ No snapshot"
                print(f"      {i+1}. Speed: {speed:.1f} km/h, Time: {ts}, {snapshot_status}")
                if snapshot:
                    exists = "✅ Exists" if os.path.exists(snapshot) else "❌ File missing"
                    print(f"         File: {snapshot} ({exists})")
        else:
            print("   ❌ Database table does not exist")
        
        conn.close()
    except Exception as e:
        print(f"   ❌ Database error: {e}")
    
    # 3. Test camera connection
    print("\n3. Testing camera connection...")
    camera_url = "http://192.168.1.109/axis-cgi/jpg/image.cgi"
    username = "root"
    password = "2024"
    
    try:
        print(f"   🌐 Testing URL: {camera_url}")
        response = requests.get(
            camera_url,
            auth=(username, password),
            timeout=10,
            verify=False
        )
        
        print(f"   📡 Response status: {response.status_code}")
        print(f"   📏 Content length: {len(response.content)} bytes")
        print(f"   📋 Content type: {response.headers.get('content-type', 'unknown')}")
        
        if response.status_code == 200:
            print("   ✅ Camera connection successful")
            
            # Try to save a test image
            test_filename = f"test_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            test_path = os.path.join(snapshots_dir, test_filename)
            
            with open(test_path, 'wb') as f:
                f.write(response.content)
            
            if os.path.exists(test_path):
                file_size = os.path.getsize(test_path)
                print(f"   ✅ Test snapshot saved: {test_path} ({file_size} bytes)")
                return True
            else:
                print(f"   ❌ Failed to save test snapshot")
                return False
        else:
            print(f"   ❌ Camera connection failed: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print("   ❌ Camera connection timeout")
        return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Camera connection error - check IP address and network")
        return False
    except Exception as e:
        print(f"   ❌ Camera test error: {e}")
        return False

def test_snapshot_function():
    """Test the snapshot capture function directly"""
    print("\n4. Testing snapshot capture function...")
    
    # Import the camera module
    try:
        from camera import capture_snapshot
        print("   ✅ Camera module imported successfully")
    except ImportError as e:
        print(f"   ❌ Failed to import camera module: {e}")
        return False
    
    # Test snapshot capture
    try:
        camera_url = "http://192.168.1.109/axis-cgi/jpg/image.cgi"
        username = "root"  
        password = "2024"
        
        print("   📸 Attempting snapshot capture...")
        snapshot_path = capture_snapshot(camera_url, "snapshots", username, password)
        
        if snapshot_path:
            print(f"   ✅ Snapshot captured: {snapshot_path}")
            if os.path.exists(snapshot_path):
                file_size = os.path.getsize(snapshot_path)
                print(f"   ✅ File exists: {file_size} bytes")
                return True
            else:
                print(f"   ❌ File path returned but file doesn't exist")
                return False
        else:
            print("   ❌ Snapshot capture returned None")
            return False
            
    except Exception as e:
        print(f"   ❌ Snapshot function error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_radar_app_logic():
    """Check the logic in radar app that should trigger snapshots"""
    print("\n5. Checking radar app logic...")
    
    # Simulate the speeding detection logic
    speed_limit = 1  # km/h from your app
    test_speeds = [0.5, 1.2, 1.5, 2.0, 0.8]
    
    print(f"   📏 Speed limit: {speed_limit} km/h")
    print("   🧪 Testing speeding detection logic:")
    
    for speed in test_speeds:
        is_speeding = speed > speed_limit
        status = "🚨 SPEEDING" if is_speeding else "✅ OK"
        print(f"      Speed: {speed:.1f} km/h -> {status}")
        
        if is_speeding:
            print(f"      🔄 This should trigger snapshot capture")

def main():
    """Main debug function"""
    print("🚗 RADAR SNAPSHOT DEBUG TOOL")
    print("=" * 60)
    
    # Run all checks
    camera_ok = check_snapshot_issue()
    snapshot_fn_ok = test_snapshot_function()
    check_radar_app_logic()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    if camera_ok and snapshot_fn_ok:
        print("✅ Camera and snapshot functions are working")
        print("🔍 Issue is likely in the radar app logic or threading")
        print("\n💡 SUGGESTED FIXES:")
        print("1. Add more debug prints in the radar app speeding detection")
        print("2. Check if the radar data processing thread is running")
        print("3. Verify that speed calculations are correct")
        print("4. Check if the queue system is working properly")
    elif camera_ok and not snapshot_fn_ok:
        print("⚠️  Camera works but snapshot function has issues")
        print("🔧 Check the camera.py module for bugs")
    elif not camera_ok:
        print("❌ Camera connection issues")
        print("🔧 Fix camera connectivity first")
        print("\n💡 CAMERA TROUBLESHOOTING:")
        print("1. Check if camera IP is correct: 192.168.1.109")
        print("2. Verify camera is powered on and connected")
        print("3. Test camera web interface in browser")
        print("4. Check username/password: root/2024")
        print("5. Try different camera URL formats")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Run this debug script to identify the exact issue")
    print("2. Check the radar app console output for error messages")
    print("3. Add temporary debug prints in the speeding detection code")
    print("4. Verify the radar data processing is actually running")

if __name__ == "__main__":
    main()