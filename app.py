from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, send_file
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import threading
import requests
from requests.auth import HTTPBasicAuth
import base64
import json
import random
import string
from urllib.parse import urlencode
import logging
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_very_strong_secret_key_here')
app.config['SESSION_COOKIE_SECURE'] = False  # Disable for local development
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session lifetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', '9cfe7dcd6b6e46aabf225e7eba73e3b4')
SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', '149d5e675a5a4bfe8ee764ab23ce20e7')
SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'https://127.0.0.1:8000/callback')
SPOTIFY_SCOPES = 'user-read-private user-read-email user-modify-playback-state user-read-playback-state streaming'

# Mood to Spotify playlist mapping
MOOD_TO_PLAYLIST = {
    'happiness': '1coQvpbb51GTntA47Hl2Su',  # Happy Hits
    'sadness': '1coQvpbb51GTntA47Hl2Su',   # Sad Songs
    'anger': '37i9dQZF1DX4sWSpwq3LiO',     # Rock Classics
    'fear': '1coQvpbb51GTntA47Hl2Su',      # Mood Booster
    'surprise': '37i9dQZF1DX4SBhb3fqCJd',   # Mood Booster
    'disgust': '37i9dQZF1DX2LTcinqsO68',    # Rock Classics
    'neutral': '37i9dQZF1DX4WYpdgoIcn6'     # Today's Top Hits
}

# Web Playback SDK configuration
SPOTIFY_PLAYER_NAME = "Emotion Music Player"
SPOTIFY_PLAYER_VOLUME = 0.5

# Emotion detection setup
model = load_model("CNN_vgg19model.h5")
emotion_labels = {
    0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness',
    4: 'sadness', 5: 'surprise', 6: 'neutral'
}
IMG_SIZE = 224
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables for emotion detection
current_emotion = "neutral"
capture_active = False
video_capture = None
frame_count = 0
captured_frames = []
processing_complete = True
last_capture_time = 0

@app.route('/login')
def login():
    state = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    session['spotify_state'] = state
    
    params = {
        'client_id': SPOTIFY_CLIENT_ID,
        'response_type': 'code',
        'redirect_uri': SPOTIFY_REDIRECT_URI,
        'scope': SPOTIFY_SCOPES,
        'state': state,
        'show_dialog': 'true'
    }
    
    auth_url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
    return redirect(auth_url)

# Helper functions
def get_auth_header():
    if 'spotify_token' not in session:
        return None
    return {'Authorization': f"Bearer {session['spotify_token']}"}

def refresh_spotify_token():
    if 'spotify_refresh_token' not in session:
        return False
    
    try:
        auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        headers = {
            'Authorization': f"Basic {auth_base64}",
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': session['spotify_refresh_token']
        }
        
        response = requests.post(
            'https://accounts.spotify.com/api/token',
            headers=headers,
            data=data,
            timeout=10
        )
        
        if response.status_code == 200:
            token_data = response.json()
            session['spotify_token'] = token_data['access_token']
            if 'refresh_token' in token_data:
                session['spotify_refresh_token'] = token_data['refresh_token']
            return True
    except Exception as e:
        logger.error(f"Token refresh failed: {str(e)}")
    return False

@app.route('/get_track_audio/<track_id>')
def get_track_audio(track_id):
    try:
        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        # First get track details to verify preview exists
        track_response = requests.get(
            f'https://api.spotify.com/v1/tracks/{track_id}',
            headers=headers,
            timeout=10
        )

        if track_response.status_code != 200:
            return jsonify(status="error", message="Track not found"), 404

        track_data = track_response.json()
        preview_url = track_data.get('preview_url')

        if not preview_url:
            return jsonify(status="error", message="No preview available"), 404

        # Stream the audio directly from Spotify
        audio_response = requests.get(preview_url, stream=True)
        if audio_response.status_code != 200:
            return jsonify(status="error", message="Preview unavailable"), 404

        # Return as a proper audio stream
        response = send_file(
            BytesIO(audio_response.content),
            mimetype='audio/mpeg',
            as_attachment=False,
            conditional=True
        )
        response.headers['Content-Length'] = len(audio_response.content)
        response.headers['Accept-Ranges'] = 'bytes'
        return response

    except Exception as e:
        logger.error(f"Audio fetch error: {str(e)}")
        return jsonify(status="error", message="Audio unavailable"), 500
@app.route('/get_spotify_token')
def get_spotify_token():
    if 'spotify_token' not in session:
        return jsonify(error="Not authenticated"), 401
    
    return jsonify(token=session['spotify_token'])
@app.route('/refresh_token')
def refresh_token_endpoint():
    if refresh_spotify_token():
        return jsonify(status="success")
    return jsonify(status="error", message="Refresh failed")

@app.route('/check_spotify_auth')
def check_spotify_auth():
    headers = get_auth_header()
    if not headers:
        return jsonify(authenticated=False)
    
    try:
        response = requests.get(
            'https://api.spotify.com/v1/me',
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            return jsonify(authenticated=True)
        
        if response.status_code == 401:
            if refresh_spotify_token():
                return jsonify(authenticated=True)
            
        return jsonify(authenticated=False)
    except Exception as e:
        logger.error(f"Auth check error: {str(e)}")
        return jsonify(authenticated=False)
    
@app.route('/transfer_playback', methods=['PUT'])  # ← Note the methods parameter
def transfer_playback():
    if 'spotify_token' not in session:
        return jsonify(error="Not authenticated"), 401

    device_id = request.json.get('device_id')
    
    response = requests.put(
        'https://api.spotify.com/v1/me/player',
        headers={
            'Authorization': f"Bearer {session['spotify_token']}",
            'Content-Type': 'application/json'
        },
        json={
            'device_ids': [device_id],
            'play': False  # Don't start playback immediately
        }
    )
    
    return ('', response.status_code)  # Return empty response with same status code
    
@app.route('/play_spotify_track', methods=['POST'])
def play_spotify_track():
    """Play a specific track with enhanced error handling"""
    if 'spotify_token' not in session:
        return jsonify(error="Not authenticated"), 401
    
    data = request.json
    headers = {
        'Authorization': f"Bearer {session['spotify_token']}",
        'Content-Type': 'application/json'
    }
    
    try:
        # First ensure shuffle is enabled
        if data.get('shuffle', True):
            shuffle_res = requests.put(
                f'https://api.spotify.com/v1/me/player/shuffle?state=true',
                headers=headers,
                params={'device_id': data.get('device_id')} if 'device_id' in data else None
            )
            if shuffle_res.status_code not in (204, 202):
                logger.warning(f"Shuffle enable failed: {shuffle_res.status_code}")
        
        # Prepare payload
        payload = {
            'uris': [data['track_uri']],
            'position_ms': 0
        }
        
        if 'device_id' in data:
            payload['device_id'] = data['device_id']
        
        # Start playback
        response = requests.put(
            'https://api.spotify.com/v1/me/player/play',
            headers=headers,
            json=payload
        )
        
        if response.status_code == 404:
            # Track not available, try to skip to next
            logger.warning(f"Track not playable: {data['track_uri']}")
            return jsonify(status="error", message="Track not playable"), 404
        
        if response.status_code == 403:
            # Premium required
            logger.warning("Premium required for playback")
            return jsonify(status="error", message="Premium account required"), 403
        
        if response.status_code not in (204, 202):
            error_msg = response.json().get('error', {}).get('message', 'Playback failed')
            logger.error(f"Playback error: {error_msg}")
            return jsonify(status="error", message=error_msg), response.status_code
        
        return jsonify(status="success")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error in play_spotify_track: {str(e)}")
        return jsonify(status="error", message="Network error"), 500
    except Exception as e:
        logger.error(f"Unexpected error in play_spotify_track: {str(e)}")
        return jsonify(status="error", message="Playback error"), 500

@app.route('/set_playback_options', methods=['PUT'])
def set_playback_options():
    """Set playback options including shuffle and context"""
    if 'spotify_token' not in session:
        return jsonify(error="Not authenticated"), 401
    
    data = request.json
    headers = {'Authorization': f"Bearer {session['spotify_token']}"}
    
    try:
        # Set shuffle state
        shuffle_state = data.get('shuffle', True)
        shuffle_res = requests.put(
            f'https://api.spotify.com/v1/me/player/shuffle?state={shuffle_state}',
            headers=headers,
            params={'device_id': data.get('device_id')} if 'device_id' in data else None
        )
        
        # If context URI is provided, set it
        if 'context_uri' in data:
            context_res = requests.put(
                'https://api.spotify.com/v1/me/player/play',
                headers=headers,
                json={'context_uri': data['context_uri']}
            )
            if context_res.status_code not in (204, 202):
                logger.error(f"Context set failed: {context_res.status_code} - {context_res.text}")
        
        return ('', shuffle_res.status_code)
    
    except Exception as e:
        logger.error(f"Playback options error: {str(e)}")
        return jsonify(error=str(e)), 500

@app.route('/play', methods=['PUT'])
def play():
    try:
        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        track_uri = request.json.get('uri')
        device_id = request.json.get('device_id')

        if not track_uri:
            return jsonify(status="error", message="Track URI required"), 400

        play_data = {'uris': [track_uri]}
        if device_id:
            play_data['device_id'] = device_id

        response = requests.put(
            'https://api.spotify.com/v1/me/player/play',
            headers=headers,
            json=play_data
        )

        if response.status_code == 204:
            return jsonify(status="success")
        return jsonify(status="error", message="Failed to start playback"), 400

    except Exception as e:
        logger.error(f"Play error: {str(e)}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/pause', methods=['PUT'])
def pause():
    try:
        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        response = requests.put(
            'https://api.spotify.com/v1/me/player/pause',
            headers=headers
        )

        if response.status_code == 204:
            return jsonify(status="success")
        return jsonify(status="error", message="Failed to pause playback"), 400

    except Exception as e:
        logger.error(f"Pause error: {str(e)}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/next', methods=['POST'])
def next_track():
    try:
        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        response = requests.post(
            'https://api.spotify.com/v1/me/player/next',
            headers=headers
        )

        if response.status_code == 204:
            return jsonify(status="success")
        return jsonify(status="error", message="Failed to skip to next track"), 400

    except Exception as e:
        logger.error(f"Next track error: {str(e)}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/previous', methods=['POST'])
def previous_track():
    try:
        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        response = requests.post(
            'https://api.spotify.com/v1/me/player/previous',
            headers=headers
        )

        if response.status_code == 204:
            return jsonify(status="success")
        return jsonify(status="error", message="Failed to go to previous track"), 400

    except Exception as e:
        logger.error(f"Previous track error: {str(e)}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/get_devices')
def get_devices():
    try:
        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        response = requests.get(
            'https://api.spotify.com/v1/me/player/devices',
            headers=headers
        )

        if response.status_code == 200:
            return jsonify(status="success", devices=response.json().get('devices', []))
        return jsonify(status="error", message="Failed to get devices"), 400

    except Exception as e:
        logger.error(f"Get devices error: {str(e)}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/get_playback_state')
def get_playback_state():
    try:
        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        response = requests.get(
            'https://api.spotify.com/v1/me/player',
            headers=headers
        )

        if response.status_code == 200:
            return jsonify(status="success", state=response.json())
        return jsonify(status="error", message="Failed to get playback state"), 400

    except Exception as e:
        logger.error(f"Get playback state error: {str(e)}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/get_playlist_tracks')
def get_playlist_tracks():
    """Get tracks for the current mood with better error handling"""
    try:
        # Check authentication first
        auth_response = check_spotify_auth()
        if not auth_response.get_json().get('authenticated'):
            return jsonify(status="error", message="Authentication required", action="reconnect"), 401

        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        emotion = current_emotion
        playlist_id = MOOD_TO_PLAYLIST.get(emotion, MOOD_TO_PLAYLIST['neutral'])

        # Get playlist details
        playlist_response = requests.get(
            f'https://api.spotify.com/v1/playlists/{playlist_id}',
            headers=headers,
            timeout=10
        )

        if playlist_response.status_code == 401:
            if refresh_spotify_token():
                headers = get_auth_header()
                playlist_response = requests.get(
                    f'https://api.spotify.com/v1/playlists/{playlist_id}',
                    headers=headers,
                    timeout=10
                )
            else:
                return jsonify(status="error", message="Token refresh failed"), 401

        if playlist_response.status_code != 200:
            error_msg = playlist_response.json().get('error', {}).get('message', 'Playlist not found')
            return jsonify(status="error", message=error_msg), playlist_response.status_code

        playlist_data = playlist_response.json()
        playlist_name = playlist_data.get('name', '')
        playlist_image = playlist_data['images'][0]['url'] if playlist_data.get('images') else ''

        # Get playlist tracks with pagination
        all_tracks = []
        next_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
        
        while next_url and len(all_tracks) < 50:  # Limit to 50 tracks
            tracks_response = requests.get(
                next_url,
                headers=headers,
                timeout=10
            )

            if tracks_response.status_code != 200:
                break

            tracks_data = tracks_response.json()
            all_tracks.extend(tracks_data.get('items', []))
            next_url = tracks_data.get('next')

        if not all_tracks:
            return jsonify(status="error", message="No tracks found"), 404

        # Process tracks with better error handling
        track_info = []
        for item in all_tracks:
            try:
                track = item.get('track')
                if not track or not track.get('id') or not track.get('is_playable', True):
                    continue
                    
                track_info.append({
                    'id': track['id'],
                    'name': track.get('name', 'Unknown Track'),
                    'artist': track['artists'][0]['name'] if track.get('artists') else 'Unknown Artist',
                    'uri': track.get('uri', ''),
                    'image': track['album']['images'][0]['url'] if track.get('album', {}).get('images') else '',
                    'duration_ms': track.get('duration_ms', 0),
                    'preview_url': track.get('preview_url')
                })
            except Exception as e:
                logger.warning(f"Skipping invalid track: {str(e)}")
                continue

        # Shuffle the tracks before returning
        random.shuffle(track_info)
        
        return jsonify(
            status="success",
            tracks=track_info[:20],  # Return max 20 tracks
            emotion=emotion,
            playlist_name=playlist_name,
            playlist_image=playlist_image
        )

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error in get_playlist_tracks: {str(e)}")
        return jsonify(status="error", message="Network error"), 500
    except Exception as e:
        logger.error(f"Unexpected error in get_playlist_tracks: {str(e)}")
        return jsonify(status="error", message="Server error"), 500

@app.route('/debug_error')
def debug_error():
    # This endpoint helps identify what's being returned when errors occur
    auth_response = check_spotify_auth()
    return jsonify({
        "auth_response_status": auth_response.status_code,
        "auth_response_content": str(auth_response.get_data()),
        "current_emotion": current_emotion,
        "session_keys": list(session.keys()),
        "spotify_token_valid": 'spotify_token' in session
    })

@app.route('/play_track', methods=['POST'])
def play_track():
    try:
        auth_response = check_spotify_auth()
        try:
            auth_data = auth_response.get_json()
        except:
            return jsonify(status="error", message="Invalid auth response format"), 500
            
        if not auth_data.get('authenticated'):
            return jsonify(status="error", message="Authentication required"), 401

        headers = get_auth_header()
        if not headers:
            return jsonify(status="error", message="Invalid token"), 401

        if not request.is_json:
            return jsonify(status="error", message="Request must be JSON"), 400

        track_uri = request.json.get('uri')
        if not track_uri:
            return jsonify(status="error", message="No track URI provided"), 400

        # First check if there's an active device
        devices_response = requests.get(
            'https://api.spotify.com/v1/me/player/devices',
            headers=headers,
            timeout=5
        )

        # Check for HTML response
        if 'Content-Type' in devices_response.headers and 'text/html' in devices_response.headers['Content-Type']:
            logger.error(f"HTML response from devices endpoint: {devices_response.text[:200]}")
            return jsonify(status="error", message="Spotify API returned unexpected response"), 500

        if devices_response.status_code == 401:
            if refresh_spotify_token():
                headers = get_auth_header()
                devices_response = requests.get(
                    'https://api.spotify.com/v1/me/player/devices',
                    headers=headers,
                    timeout=5
                )
            else:
                return jsonify(status="error", message="Token refresh failed"), 401

        if devices_response.status_code != 200:
            error_msg = f"Failed to get devices: {devices_response.status_code}"
            try:
                error_details = devices_response.json()
                error_msg = error_details.get('error', {}).get('message', error_msg)
            except:
                pass
            return jsonify(status="error", message=error_msg), devices_response.status_code

        try:
            devices = devices_response.json().get('devices', [])
        except ValueError:
            return jsonify(status="error", message="Invalid devices response"), 500

        if not devices:
            return jsonify(status="error", message="No active devices found"), 404

        # Use the first active device
        device_id = devices[0]['id']

        # Start playback
        play_response = requests.put(
            f'https://api.spotify.com/v1/me/player/play?device_id={device_id}',
            headers=headers,
            json={'uris': [track_uri]},
            timeout=10
        )

        if play_response.status_code == 204:
            return jsonify(status="success")
        else:
            error_msg = f"Playback failed: {play_response.status_code}"
            try:
                error_details = play_response.json()
                error_msg = error_details.get('error', {}).get('message', error_msg)
            except:
                pass
            return jsonify(status="error", message=error_msg), play_response.status_code

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error in play_track: {str(e)}")
        return jsonify(status="error", message=f"Network error: {str(e)}"), 500
    except Exception as e:
        logger.error(f"Unexpected error in play_track: {str(e)}")
        return jsonify(status="error", message=f"Unexpected error: {str(e)}"), 500

def is_spotify_authenticated():
    headers = get_auth_header()
    if not headers:
        return False
    try:
        response = requests.get(
            'https://api.spotify.com/v1/me',
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            return True
        if response.status_code == 401:
            if refresh_spotify_token():
                return True
        return False
    except Exception as e:
        logger.error(f"Auth check error: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/songs')
def songs():
    if 'spotify_token' not in session:
        state = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        session['spotify_state'] = state
        
        params = {
            'client_id': SPOTIFY_CLIENT_ID,
            'response_type': 'code',
            'redirect_uri': SPOTIFY_REDIRECT_URI,
            'scope': SPOTIFY_SCOPES,
            'state': state,
            'show_dialog': 'true'
        }
        
        auth_url = f"https://accounts.spotify.com/authorize?{urlencode(params)}"
        return render_template('songs.html', 
            spotify_auth_url=auth_url,
            spotify_client_id=SPOTIFY_CLIENT_ID,
            player_name=SPOTIFY_PLAYER_NAME,
            player_volume=SPOTIFY_PLAYER_VOLUME
        )
    
    return render_template('songs.html',
        spotify_client_id=SPOTIFY_CLIENT_ID,
        player_name=SPOTIFY_PLAYER_NAME,
        player_volume=SPOTIFY_PLAYER_VOLUME
    )


def camera_thread():
    """Video streaming generator function"""
    global video_capture, frame_count, captured_frames, last_capture_time, processing_complete
    
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        if capture_active and processing_complete and time.time() - last_capture_time >= 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
                face_img = face_img.astype("float32") / 255.0
                captured_frames.append(face_img)
                frame_count += 1
                logger.info(f"Captured frame {frame_count}/10")
                last_capture_time = time.time()
                
                if frame_count >= 10:
                    processing_complete = False
                    threading.Thread(target=process_frames).start()
        
        if capture_active and not processing_complete:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            text = f"Captured: {frame_count}/10"
            font_scale = frame.shape[1] / 800
            cv2.putText(frame, text, 
                       (int(frame.shape[1]/2 - 100), int(frame.shape[0]/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (255, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def process_frames():
    """Process captured frames to detect emotion"""
    global current_emotion, capture_active, frame_count, captured_frames, processing_complete
    
    try:
        logger.info("Processing captured frames...")
        X = np.array(captured_frames[:10]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        preds = model.predict(X)
        avg_pred = np.mean(preds, axis=0)
        current_emotion = emotion_labels[np.argmax(avg_pred)]
        logger.info(f"Final emotion: {current_emotion}")
        
        app.current_emotion = current_emotion
        
    except Exception as e:
        logger.error(f"Error processing frames: {str(e)}")
        current_emotion = "error"
    finally:
        captured_frames = []
        frame_count = 0
        capture_active = False
        processing_complete = True

@app.route('/verify_token')
def verify_token():
    headers = get_auth_header()
    if not headers:
        return jsonify(valid=False)
    
    try:
        response = requests.get(
            'https://api.spotify.com/v1/me',
            headers=headers,
            timeout=5
        )
        return jsonify(valid=response.status_code == 200)
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        return jsonify(valid=False)
    
@app.route('/video_feed')
def video_feed():
    return Response(camera_thread(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_capture')
def start_capture():
    global capture_active, frame_count, captured_frames, processing_complete
    
    if not capture_active and processing_complete:
        capture_active = True
        frame_count = 0
        captured_frames = []
        logger.info("Starting new capture...")
        return jsonify(status="success", message="Capture started")
    return jsonify(status="busy", message="Capture already in progress or processing")

@app.route('/get_status')
def get_status():
    return jsonify(
        capturing=capture_active,
        count=frame_count,
        emotion=current_emotion if not capture_active else "",
        processing=not processing_complete
    )

@app.route('/logout')
def logout():
    session.pop('spotify_token', None)
    session.pop('spotify_refresh_token', None)
    return redirect(url_for('songs'))

@app.route('/callback')
def callback():
    if request.args.get('state') != session.get('spotify_state'):
        return "Invalid state parameter", 400
    
    code = request.args.get('code')
    if not code:
        return "Authorization failed", 400
    
    try:
        auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
        auth_bytes = auth_string.encode('utf-8')
        auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
        
        headers = {
            'Authorization': f"Basic {auth_base64}",
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': SPOTIFY_REDIRECT_URI
        }
        
        response = requests.post(
            'https://accounts.spotify.com/api/token',
            headers=headers,
            data=data
        )
        
        if response.status_code == 200:
            token_data = response.json()
            session['spotify_token'] = token_data['access_token']
            session['spotify_refresh_token'] = token_data['refresh_token']
            session.permanent = True
            return redirect(url_for('songs'))
        return "Failed to get access token", 400
    except Exception as e:
        logger.error(f"Callback error: {str(e)}")
        return "Authentication error", 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True,
        ssl_context=('cert.pem', 'key.pem')
    )