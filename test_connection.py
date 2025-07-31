import requests

try:
    response = requests.get('http://localhost:5000', timeout=5)
    print("Server Response:")
    print(f"Status Code: {response.status_code}")
    print(f"Content: {response.text[:200]}")
except requests.exceptions.RequestException as e:
    print(f"Error connecting to server: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure the server is running in another terminal")
    print("2. Check if port 5000 is available")
    print("3. Try using a different port by modifying the URL to http://localhost:5001")
