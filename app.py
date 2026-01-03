from flask import Flask, render_template, redirect, url_for, request, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from authlib.integrations.flask_client import OAuth
import numpy as np
import cv2
import os

# TensorFlow / Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# =========================
# APP CONFIG
# =========================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

# =========================
# DATABASE
# =========================
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# =========================
# GOOGLE LOGIN
# =========================
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='ENTER_YOUR_CLIENT_ID_HERE',
    client_secret='ENTER_YOUR_CLIENT_SECRET_HERE',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile'},
)

# =========================
# DATABASE MODELS
# =========================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    doctor_name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    notes = db.Column(db.String(500))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =========================
# DOCTORS DATA
# =========================
DOCTORS_DATA = [
    {"id": 1, "name": "Dr. Arjun Verma", "specialty": "Neurologist & AI Specialist"},
    {"id": 2, "name": "Dr. Priya Sharma", "specialty": "Senior Cardiologist"},
    {"id": 3, "name": "Dr. Robert Chen", "specialty": "Orthopedics"},
    {"id": 4, "name": "Dr. Emily White", "specialty": "Pediatrician"},
    {"id": 5, "name": "Dr. Alan Grant", "specialty": "Psychiatrist"},
    {"id": 6, "name": "Dr. Sarah Connor", "specialty": "Emergency Medicine"},
]

# =========================
# AI MODEL (UNCHANGED)
# =========================
print("--- Loading Emotion Detection Model ---")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.load_weights("emotiondetector.h5")
print("--- Model Loaded Successfully ---")

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# =========================
# ROUTES
# =========================
@app.route('/')
def home():
    if not current_user.is_authenticated and 'user_name' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', doctors=DOCTORS_DATA)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('home'))
        flash("Login failed")
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    password = request.form.get('password')
    if User.query.filter_by(email=email).first():
        flash("Email already exists")
        return redirect(url_for('login'))
    user = User(email=email, password=password)
    db.session.add(user)
    db.session.commit()
    login_user(user)
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

# =========================
# GOOGLE LOGIN
# =========================
@app.route('/login/google')
def google_login():
    google = oauth.create_client('google')
    return google.authorize_redirect(url_for('google_callback', _external=True))

@app.route('/login/google/callback')
def google_callback():
    google = oauth.create_client('google')
    token = google.authorize_access_token()
    user_info = google.get('userinfo').json()
    session['user_name'] = user_info.get('name')
    session['user_email'] = user_info.get('email')
    session['user_picture'] = user_info.get('picture')
    return redirect(url_for('home'))

# =========================
# AI DETECTION (UPDATED)
# =========================
@app.route('/ai_detection')
@login_required
def ai_detection():
    return render_template('ai_detection.html')

@app.route('/predict_emotion', methods=['POST'])
@login_required
def predict_emotion():
    image_bytes = request.data
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (48,48))
    img = img / 255.0
    img = img.reshape(1,48,48,1)

    pred = model.predict(img)
    emotion = labels[int(np.argmax(pred))]
    return jsonify({"emotion": emotion})

# =========================
# OTHER PAGES
# =========================
@app.route('/doctors')
def doctors():
    return render_template('doctors.html', doctors=DOCTORS_DATA)

@app.route('/appointment', methods=['GET','POST'])
@login_required
def appointment():
    if request.method == 'POST':
        appt = Appointment(
            user_id=current_user.id,
            doctor_name=request.form.get('doctor'),
            date=request.form.get('date'),
            notes=request.form.get('notes')
        )
        db.session.add(appt)
        db.session.commit()
        flash("Appointment booked")
    return render_template('appointment.html')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/departments')
def departments(): return render_template('departments.html')

# =========================
# RUN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
