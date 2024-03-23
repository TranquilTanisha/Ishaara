import requests

def find_location_on_google_maps(latitude, longitude):
    google_maps_url = f"https://www.google.com/maps/place/{latitude},{longitude}"
    return google_maps_url

def get_current_location():
    try:
        response = requests.get('https://ipapi.co/json/')
        data = response.json()
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        return latitude, longitude
    except Exception as e:
        print("Error fetching location:", e)
        return None, None, None, None

latitude, longitude = get_current_location()
url= find_location_on_google_maps(latitude, longitude)
if latitude is not None and longitude is not None:
    print(f"Current location: Latitude - {latitude}, Longitude - {longitude}")
    print(url)
else:
    print("Failed to fetch current location.")
