from flask import Flask, flash, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import cv2
import pytesseract
import requests
from twilio.rest import Client
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize LanguageTool for grammar checking
tool = language_tool_python.LanguageTool('en-US')

# Dummy users (with WhatsApp numbers and Telegram chat IDs)
users = {
    'student': {'password': 'student123', 'phone': '+919384578453', 'telegram_chat_id': '1641632703','role':'student'},
    'teacher': {'password': 'teacher123','role':'teacher'}
}


subjects = ["Mathematics", "Physics", "Chemistry", "Biology", "Computer Science"]
evaluations = {}

# Twilio credentials for WhatsApp
TWILIO_SID = 'AC3de715b7ad0e921b0f750b543eabbfb3' 
TWILIO_AUTH_TOKEN = '359e65475eaa1c39e3abab9c28f075ff'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'

# Telegram Bot Token
TELEGRAM_BOT_TOKEN = '7769159439:AAE19l_TjMr7QFD08Bj0msWnzKh3W0Mf7s8'

# Function to send WhatsApp notification
def send_whatsapp_message(phone_number, message):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(
        body=message,
        from_=TWILIO_WHATSAPP_NUMBER,
        to=f"whatsapp:{phone_number}"
    )

# Function to send Telegram notification
def send_telegram_message(chat_id, message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(url, json=payload)

# Function to extract text from image
import google.generativeai as genai
from PIL import Image
import io
import os
from pdf2image import convert_from_path  # Converts PDF pages to images

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyA88rcBc0jfFx2eMsX0_Pqzks0KZZfI_WQ"  # Replace with your actual API key # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_image(image_path):
    """Extracts text from an image using Google Gemini API."""
    try:
        # Open the image and read its binary data
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        if not image_data:
            return "❌ Error: Image file is empty or cannot be read."

        # Validate image format
        img = Image.open(io.BytesIO(image_data))
        if img.format not in ["JPEG", "PNG"]:
            return f"❌ Error: Unsupported image format ({img.format}). Use JPEG or PNG."

        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Extract text using Gemini
        response = model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": image_data},
                "Extract only the text from the image. Do NOT summarize, explain, or interpret."
            ],
            safety_settings=[]
        )

        return response.text.strip() if response and hasattr(response, 'text') else "❌ Error: No text extracted"

    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF by converting it to images and using Gemini API."""
    try:
        poppler_path = r"C:\Users\deepa\Documents\poppler-24.08.0\Library\bin"  # Ensure path includes 'bin'

        images = convert_from_path(pdf_path, dpi=300, poppler_path=poppler_path)  # Pass poppler_path
        extracted_text = ""

        for i, image in enumerate(images):
            # Save each page as an image temporarily
            temp_image_path = f"temp_page_{i}.jpg"
            image.save(temp_image_path, "JPEG")

            # Extract text using Gemini
            extracted_text += extract_text_from_image(temp_image_path) + "\n"

            # Remove temporary image
            os.remove(temp_image_path)
        
        return extracted_text.strip()
    
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"



# Function for plagiarism detection
def detect_similarity(student_text, model_text):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([student_text, model_text])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

# Function for grammar checking
def check_grammar(text):
    matches = tool.check(text)
    return len(matches)

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username in users and users[username]['password'] == password:
        expected_role = users[username]['role']
        if request.form.get('role') != expected_role:  # Check if the selected role matches the expected role
            flash('Invalid login attempt! You are not authorized for this role.', 'error')
            return redirect(url_for('home'))
        
        session['user'] = username
        session['role'] = expected_role
        return redirect(url_for(f'{expected_role}_dashboard'))
    
    flash('Invalid credentials!', 'error')
    return redirect(url_for('home'))

@app.route('/student_dashboard')
def student_dashboard():
    if 'user' not in session or session['role'] != 'student':
        return redirect(url_for('home'))
    return render_template('student_dashboard.html', username=session['user'], subjects=subjects)

@app.route('/teacher_dashboard')
def teacher_dashboard():
    if 'user' not in session or session['role'] != 'teacher':
        return redirect(url_for('home'))
    return render_template('teacher_dashboard.html', username=session['user'], subjects=subjects)

@app.route('/upload', methods=['POST'])
def upload():
    if 'answersheet' not in request.files or 'subject' not in request.form:
        return 'No file or subject selected!'
    
    subject = request.form['subject']
    file = request.files['answersheet']
    filename = secure_filename(f"{subject}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    model_answer_path = f'model_answers/{subject}.txt'
    if not os.path.exists(model_answer_path):
        return f'Model answer for {subject} not found!'
    
    text = extract_text_from_image(filepath)
    with open(model_answer_path, 'r', encoding='utf-8', errors='ignore') as f:
     model_text = f.read().strip()

    similarity = detect_similarity(text, model_text)
    grammar_issues = check_grammar(text)
    marks = 10 if similarity >= 0.9 else 8 if similarity >= 0.7 else 5 if similarity >= 0.5 else 0
    
    evaluations[filename] = {
        'subject': subject, 
        'similarity': similarity, 
        'grammar_issues': grammar_issues, 
        'marks': marks
    }

    # Send Notifications
    student_phone = users[session['user']]['phone']
    telegram_chat_id = users[session['user']]['telegram_chat_id']
    
    message = f"Your marks for {subject}:\nSimilarity: {similarity*100:.2f}%\nGrammar Issues: {grammar_issues}\nFinal Marks: {marks}/10"
    
    send_whatsapp_message(student_phone, message)
    send_telegram_message(telegram_chat_id, message)

    return redirect(url_for('view_evaluations'))

@app.route('/upload_model_answer', methods=['POST'])
def upload_model_answer():
    if 'model_answer' not in request.files or 'subject' not in request.form:
        return 'No file or subject selected!'
    
    subject = request.form['subject']
    file = request.files['model_answer']
    os.makedirs('model_answers', exist_ok=True)
    filepath = f'model_answers/{subject}.txt'
    file.save(filepath)
    text = extract_text_from_image(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

    return f'Model answer for {subject} uploaded successfully!'

@app.route('/view_evaluations')
def view_evaluations():
    return render_template('evaluations.html', evaluations=evaluations)

@app.route('/adjust_marks', methods=['POST'])
def adjust_marks():
    filename = request.form['filename']
    new_marks = int(request.form['new_marks'])
    evaluations[filename]['marks'] = new_marks
    return redirect(url_for('view_evaluations'))

@app.route('/performance_report')
def performance_report():
    subjects = list(set(e['subject'] for e in evaluations.values()))
    avg_marks = [sum(e['marks'] for e in evaluations.values() if e['subject'] == subj) / max(1, sum(1 for e in evaluations.values() if e['subject'] == subj)) for subj in subjects]
    plt.bar(subjects, avg_marks)
    plt.xlabel("Subjects")
    plt.ylabel("Average Marks")
    plt.title("Performance Report")
    plt.savefig("static/performance.png")
    plt.close()  
    return render_template('performance.html', img_path="static/performance.png")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)