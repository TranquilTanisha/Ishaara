import requests

def check_internet_connection():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        return True  # If the request succeeds, internet connection is available
    except (requests.ConnectionError, requests.Timeout):
        return False  # If the request fails, there is no internet connection

if check_internet_connection():
    print("Internet connection is available.")
else:
    print("No internet connection.")