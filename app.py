from flask import Flask, request, jsonify, render_template, send_from_directory, redirect, url_for
from flask import session
from flask_session import Session
import subprocess
import os
import mysql.connector
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity



app = Flask(__name__)
app.secret_key = "a3b1c2d4e5f67890abcdef1234567890a1b2c3d4e5f67890abcdef1234567890"  # Replace with your strong key
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

Session(app)  # ✅ Yeh line ab work karegi
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="vsnp"
    )



@app.route("/")
def index():
    if "user_id" in session:
        return render_template("index.html")  # Agar user logged in hai toh dashboard dikhao
    return redirect(url_for("login"))  # Agar nahi toh login page bhejo


# Violations folder ko serve karne ke liye
@app.route('/violations/<path:filename>')
def serve_violations(filename):
    return send_from_directory("violations", filename)

@app.route("/process", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return jsonify({"message": "No file uploaded"}), 400

    video = request.files["video"]
    speed_limit = request.form.get("speed_limit", 80)
    real_world_distance = request.form.get("real_world_distance", 10.0)

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # Run `vsnpd.py`
    command = [
        "python", "vsnpd.py",
        "--video_path", video_path,
        "--speed_limit", str(speed_limit),
        "--real_world_distance", str(real_world_distance)
    ]

    subprocess.Popen(command)  

    return jsonify({"message": "Processing started!"})

@app.route("/get_vehicles", methods=["GET"])
def get_vehicles():
    try:
        print("🚀 Connecting to MySQL...")  # Debugging Step 1
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="1234",
            database="vsnp"
        )
        print("✅ Connected to MySQL!")  # Debugging Step 2

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Vehicles ORDER BY timestamp DESC")

        vehicles = cursor.fetchall()
        print("🚗 Vehicles Data:", vehicles)  # Debugging Step 3

        cursor.close()
        conn.close()
        return jsonify(vehicles)
    
    except Exception as e:
        print("❌ Error in get_vehicles:", str(e))  # Yeh error console me dikhega
        return jsonify({"error": str(e)}), 500

@app.route("/get_violations", methods=["GET"])
def get_violations():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="1234",
        database="vsnp"
    )
    cursor = db.cursor(dictionary=True)

    cursor.execute("SELECT * FROM Violations ORDER BY timestamp DESC")
    violations = cursor.fetchall()

    cursor.close()
    db.close()

    return jsonify(violations)

# 📝 **User Registration**
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    print("🔍 Request Form:", request.form)  # Debugging: Form fields print

    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long"}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                       (username, email, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()
        return redirect(url_for("login"))
    except mysql.connector.IntegrityError:
        return jsonify({"error": "Username or Email already exists"}), 400


# 🔑 **User Login**
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "GET":
        return render_template("login.html")  # ✅ GET request handle karo

    print("🔍 Request Form:", request.form)  # Debugging ke liye

    email = request.form.get('email')
    password = request.form.get('password')

    if not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and bcrypt.check_password_hash(user['password'], password):
        session["user_id"] = user["id"]
        return jsonify({"message": "Login successful", "redirect": url_for("index")})
    
    return jsonify({"error": "Invalid email or password"}), 401

# 🚪 **Logout**
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
