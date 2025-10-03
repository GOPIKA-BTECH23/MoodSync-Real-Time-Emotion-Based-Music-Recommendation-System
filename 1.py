import requests
import json

playlist_id = "37i9dQZF1DXdPec7aLTmlC"
headers = {
    "Authorization": "Bearer BQClfzrS8Q5rA-YfvMY_Rnthj1VGV2A-pjk8VKIsyldOf8myvrMDkmSUJQAZngY8N_vPVy3w11XvSPMczF-Scf2L8CS_N5N6ufFf1uj7C8apeggCO5Jr8VOXb7wSq_xp188XeUa9Uu-q8n1uN5SD5b_51AeMdfq4kq-ByvBzOd1lOsVMXq5TzpGAhpgG6EAtAmw2XjveCg5tXNstw5mpD9bR4TF7HGA7mdIohTr5RWF-NitCPhOci4EY3aLYsqNeC1zVjw4NtQm5"  # Put your real token here
}

response = requests.get(
    f"https://api.spotify.com/v1/playlists/{playlist_id}",
    headers=headers
)

print("Status Code:", response.status_code)
print("Response Text:", json.dumps(response.json(), indent=2))
