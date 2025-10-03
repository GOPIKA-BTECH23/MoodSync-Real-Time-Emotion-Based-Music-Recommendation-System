# MoodSync-Real-Time-Emotion-Based-Music-Recommendation-System
MoodSync is an intelligent, real-time web application that creates a personalized music listening experience by detecting a user's current facial emotion and automatically curating and playing a mood-matching playlist from Spotify. It seamlessly blends 
Computer Vision, Deep Learning, and the Spotify Web API to deliver context-aware music recommendations.

💡 Project Overview

Most existing music recommendation systems rely on static data like listening history or fixed preferences, ignoring the user's current emotional state. MoodSync solves this by detecting emotions in real-time through facial expressions and intelligently recommending music that aligns with or enhances the user’s mood, providing a more responsive and personalized listening experience.

Key Features

Real-Time Emotion Detection: Captures live video via webcam using OpenCV and analyzes faces using Haar Cascade to identify one of seven emotions (happy, sad, angry, neutral, surprise, disgust, fear).


Deep Learning Model: Utilizes a pre-trained VGG19 Convolutional Neural Network (CNN) , trained on the 
FER-2013 dataset , for high-accuracy facial expression recognition (93% on test data).

Seamless Spotify Integration: Authenticates the user via OAuth 2.0 and uses the Spotify Web API and SDK to fetch mood-matching playlists and stream music directly to the user's active device.

Performance: Achieves low end-to-end latency of approximately 150ms  for a lag-free experience.

Tech Stack: Built with Python, TensorFlow, and Flask.

⚙️ Project Structure

The project follows a standard Flask application structure:
<img width="904" height="529" alt="image" src="https://github.com/user-attachments/assets/42b4e781-f92b-48fa-815f-fbc079c56360" />

🚀 How to Run MoodSync Locally (Complete Workflow)

Follow these steps precisely to set up and launch the MoodSync web application on your local machine.

Step 1: Prerequisites
Python: Ensure you have Python 3.8+ installed.

Webcam: An integrated or external webcam is required.

Spotify API Keys:

Go to the Spotify Developer Dashboard.

Create a new application to get your Client ID and Client Secret.

In the app settings, add http://127.0.0.1:5000/callback as a Redirect URI.

Step 2: Setup and Installation

1.Clone the Repository:
git clone [Your-GitHub-Repo-URL]
cd MoodSync
# Navigate to the website folder where your app.py is located
# cd Summer_Internship-2/Website/Website

2.Create a Virtual Environment (Highly Recommended):
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows

3.Install Python Dependencies:
Install all required libraries using the provided 
requirements.txt file.
pip install -r requirements.txt

Step 3: Configure Environment Variables
The application needs your Spotify credentials to handle the OAuth process. Set these in your terminal session.
<img width="834" height="284" alt="image" src="https://github.com/user-attachments/assets/8a4ed9f3-c45d-4fe7-9230-3dd1ad62a8cc" />

Step 4: Run the Application
Execute the Flask server from your terminal:
flask run

Step 5: Access and Use
1.Open the Browser: The terminal will output a URL, typically http://127.0.0.1:5000/. Open this in your web browser.

2.Connect to Spotify: The application will prompt you to log in to Spotify and grant permissions (OAuth 2.0 flow).

3.Detect Mood: Click the "Detect Mood" button to start the webcam feed and emotion analysis.

4.Get Music: The system will display the detected emotion (e.g., "Detected Mood: happiness") and automatically fetch and play a matching Spotify playlist (e.g., "Happy Hits") on your active Spotify device.

🔮 Future Enhancements

Multi-User Support: Allow simultaneous mood detection and personalized playlists for multiple users.

Improved Emotion Categories: Expand the VGG19 model to detect subtle emotions like calm or anxious for more granular recommendations.

Offline Mode: Implement local caching of Spotify playlists to reduce API dependency and improve loading speed.

Multi-Modal AI: Integrate voice sentiment analysis or text inputs for a more robust hybrid recommendation system.
<img width="1309" height="620" alt="image" src="https://github.com/user-attachments/assets/c5efec39-2be2-4c78-b568-ba6f2d13e6ac" />

<img width="1262" height="595" alt="image" src="https://github.com/user-attachments/assets/9b0f6a30-446e-4a8a-91c5-aafd915da045" />

