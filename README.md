# CheatGuard
CheatGuard is a lightweight Python-based exam proctoring system designed for offline schools that don’t have access to expensive AI infrastructure or online proctoring tools. It uses OpenCV + MediaPipe to monitor students through a camera and detect suspicious behavior like:
* Looking away/down repeatedly
* Hands appearing in frame
* Absence of face during the exam
## Features
* Live video monitoring via webcam
* Face detection to ensure student presence
<!-- * Hand detection to detect possible cheating gestures
* Head/face orientation checks (looking down / away) -->
* Automatic evidence capture when suspicious activity is detected
* Logs alerts to a Flask-SQLAlchemy database with timestamps and photographical evidence
* Built fully using classical CV + MediaPipe (no ML training required)
## Tech Stack
* Python 3
* OpenCV
* MediaPipe
* NumPy
# How to install
1. Ensure that [Python 3.12](https://www.python.org/downloads/release/python-31210/) is installed. 
2. Clone the repository with:
   ```
   git clone https://github.com/muditgoel135/CheatGuard.git
   cd CheatGuard
   ```
3. Create a venv and activate it in your terminal.
4. Download all the dependencies from requirements.txt using the below command.
	```
   pip install -r requirements.txt
 	```
5. Decide on a secret key to use. 
6. Run the app with the below command:
	```
	python app.py your_secret_key
	```
# How It Works
1. Webcam feed is captured using OpenCV
2. MediaPipe processes each frame in real-time
3. CheatGuard checks for:
	* Face presence
	<!-- * Hand visibility
	* Suspicious head orientation -->
4. If a rule is violated
	1. An alert is logged.
	 2. It is reported to the invigilator.
	 3. A frame is saved as evidence
# Design Goals
1. Teacher-centric → Final desicions are made by the invigilator
2. Academic-focused → Built for real exam environments
3. Transparent logic → Rule-based detection ensures alerts are interpretable, not black-box
4. Affordable → Works with cheap webcams to support budget-constrained schools
5. Offline-first → Designed for schools without online infrastructure
## Current Limitations
* Multi-camera support is not tested
* FPS depends on webcam + system performance
* Not intended to replace human invigilators (yet 👀)
## Future Improvements
* Hand raise detection
* Multi-camera classroom support
* Student ID verification
* Gaze estimation
* GUI for invigilators
* Performance optimizations for low-end systems
# Contributing
This project was built fast and intentionally simple.
If you have ideas, optimizations, or improvements — PRs are welcome.
# License
This project is licensed under the MIT License.
Do whatever you want, just don’t blame me if students outsmart it 😄
