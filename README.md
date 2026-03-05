# CheatGuard
CheatGuard is a lightweight Python-based exam proctoring system designed for offline schools that don’t have access to expensive AI infrastructure or online proctoring tools. It uses OpenCV + MediaPipe to monitor students through a camera and detect suspicious behaviour like:
* Looking away/down repeatedly
* Hands appearing in frame
* Absence of face during the exam
* Detection of prohibited items during the exam
## Current Features
* Live video monitoring via webcam
* Face detection to ensure student presence
* Hand detection for questions during the exam
* Automatic photographic and videographic evidence capture when suspicious activity is detected
* Logs alerts to a Flask-SQLAlchemy database with timestamps and photographic evidence
* Saves videographic evidence with timestamps locally
* Built fully using classical CV + MediaPipe (no ML training required)
## Tech Stack
* Python 3
* OpenCV
* MediaPipe
* NumPy
# How to install
1. Ensure that [Python 3.12](https://www.python.org/downloads/release/python-31210/) is installed. **Note**: The program will **NOT** work on a later version of Python.
2. Clone the repository by running the command below in your terminal:
   ```
   git clone https://github.com/muditgoel135/CheatGuard.git
   cd CheatGuard
   ```
3. Create a venv and activate it in your terminal.
4. Download all the dependencies from requirements.txt using the command below.
	```
    pip install -r requirements.txt
 	```
5. Create a `.env` file with your secret key with the name `SECRET_KEY`. Save it as the text below and replace "your_secret_key" with your secret key.
   ```
   SECRET_KEY = your_secret_key
   ```
7. Run the app with the following command:
	```
	python app.py
	```
# How It Works
1. Webcam feed is captured using OpenCV
2. MediaPipe processes each frame in real-time
3. CheatGuard checks for:
	* Face presence
	* Hand visibility
4. The program draws MediaPipe's given face and hand landmarks
5. If a rule is violated:
	1. An alert is logged.
	 2. It is reported to the invigilator.
	 3. A frame is saved as evidence
# Design Goals
1. **Teacher-centric**: Final decisions are made by the invigilator
2. **Academic-focused**: Built for real exam environments
3. **Transparent logic**: Rule-based detection ensures alerts are interpretable, not black-box
4. **Affordable**: Works with cheap webcams to support budget-constrained schools
5. **Offline-first**: Designed for schools without online infrastructure
## Current Limitations
* Multi-camera support is not tested
* FPS depends on webcam + system performance
* Lack of simplicity in UI and UX for non-techy invigilators
* Not intended to replace human invigilators (yet 👀)
## Future Improvements
* Multi-camera classroom support
* Student ID verification
* Gaze estimation
* GUI for invigilators
* Performance optimisations for low-end systems
* Object detection
# Contributing
This project was built fast and intentionally simple.
If you have ideas, optimisations, or improvements, PRs are welcome.
# License
This project is licensed under the MIT License.
