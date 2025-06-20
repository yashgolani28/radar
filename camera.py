import requests
import os
from datetime import datetime
import time
import urllib3
from requests.auth import HTTPDigestAuth

# Disable SSL warnings for local cameras
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def capture_snapshot(camera_url, output_dir="snapshots", username=None, password=None, timeout=10):
    try:
        if not os.path.exists(output_dir):
            print(f"[CAMERA] Creating directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        print(f"[CAMERA] Requesting snapshot from {camera_url}")

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'image/jpeg,image/png,image/*,*/*',
            'Connection': 'close'
        }

        response = requests.get(
            camera_url,
            auth=HTTPDigestAuth(username, password) if username and password else None,
            timeout=timeout,
            headers=headers,
            verify=False,
            stream=True
        )

        print(f"[CAMERA] Response status: {response.status_code}")
        print(f"[CAMERA] Content-Type: {response.headers.get('content-type', 'unknown')}")
        print(f"[CAMERA] Content-Length: {response.headers.get('content-length', 'unknown')}")

        if response.status_code == 200:
            content_type = response.headers.get('content-type', '').lower()
            if 'image' not in content_type:
                print(f"[CAMERA] Warning: Unexpected content type: {content_type}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"speeding_{timestamp}.jpg")

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"[CAMERA] Snapshot saved: {filename} ({file_size} bytes)")

                if file_size == 0:
                    print(f"[CAMERA] ERROR: File is empty, removing...")
                    os.remove(filename)
                    return None
                elif file_size < 1024:
                    print(f"[CAMERA] WARNING: File is very small ({file_size} bytes)")

                return filename
            else:
                print(f"[CAMERA] ERROR: File was not created")
                return None

        elif response.status_code == 401:
            print(f"[CAMERA] ERROR: Authentication failed (401) - check username/password")
        elif response.status_code == 404:
            print(f"[CAMERA] ERROR: Camera URL not found (404) - check URL path")
        elif response.status_code == 403:
            print(f"[CAMERA] ERROR: Access forbidden (403) - check permissions")
        else:
            print(f"[CAMERA] ERROR: HTTP {response.status_code} - {response.reason}")
            print(f"[CAMERA] Response: {response.text[:200]}...")

    except requests.exceptions.Timeout:
        print(f"[CAMERA] ERROR: Request timeout after {timeout} seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"[CAMERA] ERROR: Connection failed - {e}")
        print(f"[CAMERA] Check if camera IP {camera_url} is reachable")
    except requests.exceptions.RequestException as e:
        print(f"[CAMERA] ERROR: Request failed - {e}")
    except PermissionError:
        print(f"[CAMERA] ERROR: Permission denied writing to {output_dir}")
    except OSError as e:
        print(f"[CAMERA] ERROR: OS error - {e}")
    except Exception as e:
        print(f"[CAMERA] ERROR: Unexpected error - {e}")
        import traceback
        traceback.print_exc()

    return None

def test_camera_connection(camera_url, username=None, password=None):
    try:
        print(f"[CAMERA TEST] Testing connection to {camera_url}")

        response = requests.get(
            camera_url,
            auth=HTTPDigestAuth(username, password) if username and password else None,
            timeout=5,
            verify=False
        )

        if response.status_code == 200:
            content_length = len(response.content)
            content_type = response.headers.get('content-type', 'unknown')
            print(f"[CAMERA TEST] SUCCESS: {content_length} bytes, type: {content_type}")
            return True
        else:
            print(f"[CAMERA TEST] FAILED: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"[CAMERA TEST] FAILED: {e}")
        return False