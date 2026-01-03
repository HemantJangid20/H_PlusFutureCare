from flask import Flask, render_template, Response, redirect, url_for, request, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from authlib.integrations.flask_client import OAuth
import cv2
import numpy as np
import os

# Import Keras layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secretkey123' # Change this to a random long string in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

# --- DATABASE SETUP ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- GOOGLE LOGIN SETUP ---
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id='ENTER_YOUR_CLIENT_ID_HERE',         # <--- PASTE FROM GOOGLE CONSOLE
    client_secret='ENTER_YOUR_CLIENT_SECRET_HERE', # <--- PASTE FROM GOOGLE CONSOLE
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'openid email profile'},
)

# --- DATABASE MODELS ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False) # Saves who booked it
    doctor_name = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(50), nullable=False)
    notes = db.Column(db.String(500))

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- DOCTORS DATA (For Search & Display) ---
DOCTORS_DATA = [
    {
        "id": 1, 
        "name": "Dr. Arjun Verma", 
        "specialty": "Neurologist & AI Specialist", 
        "description": "Expert in brain mapping and using our Emotion AI.",
        "image": "👨‍⚕️", 
        "color": "blue" 
    },
    {
        "id": 2, 
        "name": "Dr. Priya Sharma", 
        "specialty": "Senior Cardiologist", 
        "description": "Specializes in non-invasive heart surgeries.",
        "image": "👩‍⚕️", 
        "color": "purple"
    },
    {
        "id": 3, 
        "name": "Dr. Robert Chen", 
        "specialty": "Head of Orthopedics", 
        "description": "Pioneering new techniques in joint replacement.",
        "image": "👨‍⚕️", 
        "color": "green"
    },
    {
        "id": 4, 
        "name": "Dr. Emily White", 
        "specialty": "Pediatrician", 
        "description": "Dedicated to child wellness and early screening.",
        "image": "👩‍⚕️", 
        "color": "pink"
    },
    {
        "id": 5, 
        "name": "Dr. Alan Grant", 
        "specialty": "Psychiatrist", 
        "description": "Integrating traditional therapy with emotion tracking.",
        "image": "👨‍⚕️", 
        "color": "yellow"
    },
    {
        "id": 6, 
        "name": "Dr. Sarah Connor", 
        "specialty": "Emergency Medicine", 
        "description": "Expert in trauma care and rapid response.",
        "image": "👩‍⚕️", 
        "color": "red"
    }
]

# --- AI & CAMERA SETUP ---
if os.path.exists('haarcascade_frontalface_default.xml'):
    haar_path = 'haarcascade_frontalface_default.xml'
else:
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
face_cascade = cv2.CascadeClassifier(haar_path)

print("--- Building Model & Loading Weights... ---")
try:
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])
    model.load_weights("emotiondetector.h5")
    print("--- Model Loaded Successfully! ---")
except Exception as e:
    print(f"!!! ERROR LOADING AI MODEL: {e} !!!")
    model = None

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (p, q, r, s) in faces:
                cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
                if model:
                    try:
                        image = gray[q:q+s, p:p+r]
                        image = cv2.resize(image, (48, 48))
                        img = extract_features(image)
                        pred = model.predict(img)
                        prediction_label = labels[pred.argmax()]
                        cv2.putText(frame, prediction_label, (p-10, q-10), 
                                   cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
                    except Exception:
                        pass
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- ROUTES ---

@app.route('/')
def home():
    # Pass doctors data to index.html for the search bar autosuggest
    if not current_user.is_authenticated and 'user_name' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', doctors=DOCTORS_DATA)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Login Failed.')
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    password = request.form.get('password')
    if User.query.filter_by(email=email).first():
        flash('Email exists.')
        return redirect(url_for('login'))
    new_user = User(email=email, password=password)
    db.session.add(new_user)
    db.session.commit()
    login_user(new_user)
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    # Clear both Flask-Login and Google Session
    logout_user()
    session.clear()
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        print(f"Password reset requested for: {email}")
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

# --- SOCIAL LOGIN ROUTES ---

@app.route('/login/google')
def google_login():
    google = oauth.create_client('google')
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def google_callback():
    google = oauth.create_client('google')
    token = google.authorize_access_token()
    user_info = google.get('userinfo').json()
    
    # Save to Session
    session.get['user_name'] = user_info.get('name')
    session.get['user_picture'] = user_info.get('picture')
    session.get['user_email'] = user_info.get('email')
    
    return redirect(url_for('home')) # Redirect to home/dashboard

@app.route('/login/apple')
def apple_login():
    return "Apple Login requires a Developer Account. Coming soon!", 200

# --- MAIN FEATURES ---

@app.route('/search')
def search():
    query = request.args.get('query', '').lower()
    
    # Filter doctors based on name or specialty
    results = [
        doc for doc in DOCTORS_DATA 
        if query in doc['name'].lower() or query in doc['specialty'].lower()
    ]
    
    # Show results on the doctors page
    return render_template('doctors.html', doctors=results, search_query=query)

@app.route('/doctors')
def doctors(): 
    # Show all doctors
    return render_template('doctors.html', doctors=DOCTORS_DATA)

@app.route('/appointment', methods=['GET', 'POST'])
@login_required
def book_appointment():
    selected_doctor = request.args.get('doctor', '') 
    
    if request.method == 'POST':
        doctor_name = request.form.get('doctor')
        date_str = request.form.get('date')
        notes = request.form.get('notes')
        
        new_appt = Appointment(
            user_id=current_user.id,
            doctor_name=doctor_name,
            date=date_str,
            notes=notes
        )
        db.session.add(new_appt)
        db.session.commit()
        
        flash('Appointment Booked Successfully!')
        return redirect(url_for('book_appointment'))

    return render_template('appointment.html', selected_doctor=selected_doctor)

@app.route('/ai_detection')
@login_required
def ai_detection():
    return render_template('detection.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/departments')
def departments(): return render_template('departments.html')

@app.route('/dashboard')
def dashboard():
    if not current_user.is_authenticated and 'user_name' not in session:
        return redirect(url_for('login'))

    return render_template(
        'dashboard.html',
        name=session.get('user_name', 'User'),
        picture=session.get('user_picture', None)
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)