import requests
import base64

# Fill these in
CLIENT_ID = '9cfe7dcd6b6e46aabf225e7eba73e3b4'
CLIENT_SECRET = '149d5e675a5a4bfe8ee764ab23ce20e7'
REDIRECT_URI = 'https://127.0.0.1:8000/callback'
AUTH_CODE = 'AQCLXddk11WeTwwdLv29J_wYVaVAJ3cy-4KQnJmeAD3NsJQ4F8zdna6pFDmOSfdtWWEa-EqKD7t1BE7p8GRmYR53-q_minH6ILxfxP2Rw-ho4tBZTvyUd7eWSbhFRk_9X4E_bQtUie5GUx78oWh4lFsK4VIo0rtnzv3R6q0KX1CLxXZjvqZoIXGAnmdhaL-Y2yOrzk4u8ygGdyWX_JK9cgG_OlsrcCss4abLr29y-jbMFx1wi13xqYltlHvBMv0nVkCHg-thhUWczfOwOuMj2SN68_zsyF4XYlNA'  # You get this after user authorizes

def get_access_token(client_id, client_secret, code, redirect_uri):
    auth_str = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()

    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f'Basic {b64_auth_str}',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': redirect_uri
    }

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        tokens = response.json()
        print('Access Token:', tokens['access_token'])
        print('Refresh Token:', tokens['refresh_token'])
        return tokens
    else:
        print('Failed to get token:', response.status_code, response.text)
        return None

# Example usage:
get_access_token(CLIENT_ID, CLIENT_SECRET, AUTH_CODE, REDIRECT_URI)
