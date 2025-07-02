import streamlit as st
import os
import re
import tempfile
from pypdf import PdfReader
from pinecone import Pinecone
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize session state
if 'quiz_generated' not in st.session_state:
    st.session_state.quiz_generated = False
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'student_info' not in st.session_state:
    st.session_state.student_info = None
# Track quiz attempts (max 2)
if 'quiz_attempts' not in st.session_state:
    st.session_state.quiz_attempts = 0
# Timer state
if 'quiz_start_time' not in st.session_state:
    st.session_state.quiz_start_time = None

# Gemini API helpers
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={GEMINI_API_KEY}"
GEMINI_FLASH_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

def get_gemini_embedding(text):
    headers = {"Content-Type": "application/json"}
    data = {"content": {"parts": [{"text": text}]}}
    try:
        response = requests.post(GEMINI_EMBED_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["embedding"]["values"]
        else:
            st.error(f"Gemini embedding failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return None

def extract_json(text):
    # Remove Markdown code block markers if present
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    # Extract the first {...} block from the text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        json_str = match.group(0)
        # Remove trailing commas before } or ]
        json_str = re.sub(r',([ \t\r\n]*[}\]])', r'\1', json_str)
        try:
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Quiz JSON parsing error: {e}")
            return None
    else:
        st.error("No JSON found in Gemini output.")
        return None

def generate_quiz(text):
    prompt = f"""
You are a quiz generator. Given the following class notes, generate a quiz with:
- 5 MCQs (1 mark each, one correct answer)
- 2 descriptive questions (3 marks each)
- 2 true/false questions (1 mark each)
- 1 MCQ with multiple correct answers (2 marks)

Ensure all important concepts are covered, and difficulty is balanced. Return ONLY valid JSON in this exact format:
{{
  "mcq": [
    {{"question": "Question text", "options": ["A", "B", "C", "D"], "answer": "Correct option"}},
    {{"question": "Question text", "options": ["A", "B", "C", "D"], "answer": "Correct option"}},
    {{"question": "Question text", "options": ["A", "B", "C", "D"], "answer": "Correct option"}},
    {{"question": "Question text", "options": ["A", "B", "C", "D"], "answer": "Correct option"}},
    {{"question": "Question text", "options": ["A", "B", "C", "D"], "answer": "Correct option"}}
  ],
  "descriptive": [
    {{"question": "Question text", "answer": "Expected answer"}},
    {{"question": "Question text", "answer": "Expected answer"}}
  ],
  "true_false": [
    {{"question": "Statement", "answer": true}},
    {{"question": "Statement", "answer": false}}
  ],
  "multi_mcq": {{
    "question": "Question text",
    "options": ["A", "B", "C", "D"],
    "answers": ["Correct option 1", "Correct option 2"]
  }}
}}

Class notes:
{text[:4000]}
"""
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048}
    }
    
    try:
        response = requests.post(GEMINI_FLASH_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            quiz_json = extract_json(content)
            return quiz_json
        else:
            st.error(f"Gemini quiz generation failed: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return None

def grade_descriptive(answer, reference):
    if not answer or not answer.strip():
        return 0
    
    prompt = f"""
Compare the student answer with the reference answer and give a score out of 3.
Consider:
- Accuracy of information (1 mark)
- Completeness (1 mark)
- Clarity and structure (1 mark)

Reference answer: {reference}
Student answer: {answer}

Return ONLY a number between 0 and 3.
"""
    
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 10}
    }
    
    try:
        response = requests.post(GEMINI_FLASH_URL, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            score = float(re.search(r'\d+\.?\d*', content).group())
            return min(max(int(score), 0), 3)
        else:
            return 0
    except:
        return 0

def send_gmail(to_email, subject, body):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    from_email = os.getenv("SMTP_EMAIL")
    from_password = os.getenv("SMTP_PASSWORD")
    
    if not from_email or not from_password:
        st.error("SMTP credentials not set in .env file.")
        return False
    
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))
    
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# Initialize Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "quiz-notes"
    
    # Check if index exists
    try:
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(name=index_name, dimension=768, metric='cosine')
        index = pc.Index(index_name)
    except Exception as e:
        st.error(f"Pinecone initialization error: {e}")
        index = None
else:
    st.error("PINECONE_API_KEY not found in environment variables")
    index = None

# Streamlit UI
st.set_page_config(page_title="AI Quiz Generator", page_icon="📝", layout="wide")
st.title("🎓 AI Quiz Generator for Students")

# Student Information Form and Logout Option
if not st.session_state.student_info:
    st.markdown("### 👤 Student Information")
    with st.form("student_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Student Name", placeholder="Enter your full name")
            email = st.text_input("Email", placeholder="your.email@example.com")
        with col2:
            roll_no = st.text_input("Roll Number", placeholder="Enter your roll number")
        submitted = st.form_submit_button("Start Quiz", use_container_width=True)
        if submitted:
            if name and roll_no and email:
                st.session_state.student_info = {
                    "name": name, 
                    "roll_no": roll_no, 
                    "email": email,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.rerun()
            else:
                st.error("Please fill all fields")



# Show logout button and welcome bar at the top right if logged in, but do NOT block the rest of the quiz logic
if st.session_state.student_info:
    col_welcome, col_logout = st.columns([6, 1])
    with col_welcome:
        st.info(f"Welcome, {st.session_state.student_info['name']}! 🎯")
    with col_logout:
        if st.button("🚪 Log Out", key="logout_btn", use_container_width=True):
            # Reset all session state except attempts
            st.session_state.student_info = None
            st.session_state.quiz_generated = False
            st.session_state.quiz_submitted = False
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score = None
            st.session_state.desc_scores = []
            st.session_state.quiz_start_time = None
            st.rerun()

# Generate Quiz (only allow if attempts < 2)
if (
    st.session_state.student_info
    and not st.session_state.quiz_generated
    and not st.session_state.quiz_submitted
):
    # Check attempts
    attempts_left = 2 - st.session_state.quiz_attempts
    if attempts_left <= 0:
        st.error("You have reached the maximum number of quiz attempts (2). Please contact your teacher if you need more attempts.")
    else:
        st.warning(f"Attempts left: {attempts_left} (Max 2)")
        # Load teacher notes
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📚 Load Quiz from Teacher Notes", use_container_width=True):
                with st.spinner("🔄 Generating quiz from teacher notes..."):
                    notes_path = "teacher_notes.txt"
                    if os.path.exists(notes_path):
                        with open(notes_path, "r", encoding="utf-8") as f:
                            text = f.read()
                        quiz = generate_quiz(text)
                        if quiz:
                            st.session_state.quiz_data = quiz
                            st.session_state.quiz_generated = True
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_start_time = datetime.now().timestamp()
                            st.success("✅ Quiz generated successfully! You have 20 minutes to complete the quiz.")
                            st.rerun()
                        else:
                            st.error("Failed to generate quiz. Please try again.")
                    else:
                        st.error("❌ Teacher notes not found. Please contact your teacher.")

# Display and Handle Quiz (with timer and attempt limit)
elif st.session_state.quiz_generated and not st.session_state.quiz_submitted:
    # Timer logic
    import time
    quiz_duration_sec = 20 * 60  # 20 minutes
    start_time = st.session_state.quiz_start_time
    now = datetime.now().timestamp()
    time_left = int(quiz_duration_sec - (now - start_time)) if start_time else quiz_duration_sec
    minutes, seconds = divmod(max(time_left, 0), 60)
    st.markdown(f"### 📝 Quiz for {st.session_state.student_info['name']} (Roll No: {st.session_state.student_info['roll_no']})")
    st.markdown(f"**⏰ Time left: {minutes:02d}:{seconds:02d}**  (20:00 max)")
    st.markdown("---")
    quiz = st.session_state.quiz_data
    # If time is up, auto-submit
    if time_left <= 0:
        st.warning("Time is up! Your quiz is being submitted automatically.")
        st.session_state.quiz_submitted = True
        st.rerun()
    else:
        with st.form("quiz_form"):
            # MCQs (5 questions, 1 mark each)
            st.markdown("### 🔘 Multiple Choice Questions (1 mark each)")
            for i, q in enumerate(quiz.get("mcq", [])):
                st.markdown(f"**Q{i+1}.** {q['question']}")
                answer = st.radio(
                    label="Select one:",
                    options=q['options'],
                    key=f"mcq_{i}",
                    label_visibility="collapsed"
                )
                st.session_state.quiz_answers[f"mcq_{i}"] = answer
                st.markdown("---")
            # Descriptive Questions (2 questions, 3 marks each)
            st.markdown("### ✍️ Descriptive Questions (3 marks each)")
            for i, q in enumerate(quiz.get("descriptive", [])):
                st.markdown(f"**Q{i+6}.** {q['question']}")
                answer = st.text_area(
                    label="Your answer:",
                    key=f"desc_{i}",
                    height=100,
                    label_visibility="collapsed"
                )
                st.session_state.quiz_answers[f"desc_{i}"] = answer
                st.markdown("---")
            # True/False (2 questions, 1 mark each)
            st.markdown("### ✓✗ True/False Questions (1 mark each)")
            for i, q in enumerate(quiz.get("true_false", [])):
                st.markdown(f"**Q{i+8}.** {q['question']}")
                answer = st.radio(
                    label="Select one:",
                    options=["True", "False"],
                    key=f"tf_{i}",
                    label_visibility="collapsed"
                )
                st.session_state.quiz_answers[f"tf_{i}"] = answer
                st.markdown("---")
            # Multi-correct MCQ (1 question, 2 marks)
            st.markdown("### 🔲 Multiple Correct Answers (2 marks)")
            if "multi_mcq" in quiz:
                q = quiz["multi_mcq"]
                st.markdown(f"**Q10.** {q['question']} *(Select all that apply)*")
                answer = st.multiselect(
                    label="Select all correct options:",
                    options=q['options'],
                    key="multi_mcq",
                    label_visibility="collapsed"
                )
                st.session_state.quiz_answers["multi_mcq"] = answer
            st.markdown("---")
            submit_quiz = st.form_submit_button("🚀 Submit Quiz", use_container_width=True)
            if submit_quiz:
                st.session_state.quiz_submitted = True
                st.rerun()

elif st.session_state.quiz_submitted:
    st.markdown("### 📊 Quiz Results")
    
    quiz = st.session_state.quiz_data
    answers = st.session_state.quiz_answers
    
    # Calculate scores
    total_score = 0
    mcq_score = 0
    desc_scores = []
    tf_score = 0
    multi_score = 0
    
    # Create detailed results
    results_detail = []
    
    # Grade MCQs (5 questions, 1 mark each)
    st.markdown("#### Multiple Choice Questions")
    for i, q in enumerate(quiz.get("mcq", [])):
        user_answer = answers.get(f"mcq_{i}", "")
        correct_answer = q.get("answer", "")
        is_correct = user_answer == correct_answer
        
        if is_correct:
            mcq_score += 1
            total_score += 1
            status = "✅"
        else:
            status = "❌"
        
        results_detail.append({
            "Question": f"Q{i+1}",
            "Type": "MCQ",
            "Your Answer": user_answer,
            "Correct Answer": correct_answer,
            "Status": status,
            "Marks": "1/1" if is_correct else "0/1"
        })
    
    st.success(f"MCQ Score: {mcq_score}/5")
    
    # Grade Descriptive Questions (2 questions, 3 marks each)
    st.markdown("#### Descriptive Questions")
    for i, q in enumerate(quiz.get("descriptive", [])):
        user_answer = answers.get(f"desc_{i}", "")
        reference_answer = q.get("answer", "")
        
        with st.spinner(f"Grading descriptive question {i+1}..."):
            marks = grade_descriptive(user_answer, reference_answer)
        
        desc_scores.append(marks)
        total_score += marks
        
        results_detail.append({
            "Question": f"Q{i+6}",
            "Type": "Descriptive",
            "Your Answer": user_answer[:100] + "..." if len(user_answer) > 100 else user_answer,
            "Reference": reference_answer[:100] + "..." if len(reference_answer) > 100 else reference_answer,
            "Status": "🔍",
            "Marks": f"{marks}/3"
        })
    
    desc_total = sum(desc_scores)
    st.success(f"Descriptive Score: {desc_total}/6 (Q6: {desc_scores[0]}/3, Q7: {desc_scores[1]}/3)")
    
    # Grade True/False (2 questions, 1 mark each)
    st.markdown("#### True/False Questions")
    for i, q in enumerate(quiz.get("true_false", [])):
        user_answer = answers.get(f"tf_{i}", "")
        correct_answer = str(q.get("answer", "")).capitalize()
        is_correct = user_answer == correct_answer
        
        if is_correct:
            tf_score += 1
            total_score += 1
            status = "✅"
        else:
            status = "❌"
        
        results_detail.append({
            "Question": f"Q{i+8}",
            "Type": "True/False",
            "Your Answer": user_answer,
            "Correct Answer": correct_answer,
            "Status": status,
            "Marks": "1/1" if is_correct else "0/1"
        })
    
    st.success(f"True/False Score: {tf_score}/2")
    
    # Grade Multi-correct MCQ (1 question, 2 marks)
    st.markdown("#### Multiple Correct Answers")
    if "multi_mcq" in quiz:
        user_answer_multi = set(answers.get("multi_mcq", []))
        correct_answer_multi = set(quiz["multi_mcq"].get("answers", []))
        is_correct = user_answer_multi == correct_answer_multi
        
        if is_correct:
            multi_score = 2
            total_score += 2
            status = "✅"
        else:
            status = "❌"
        
        results_detail.append({
            "Question": "Q10",
            "Type": "Multi-MCQ",
            "Your Answer": ", ".join(sorted(user_answer_multi)) if user_answer_multi else "None",
            "Correct Answer": ", ".join(sorted(correct_answer_multi)),
            "Status": status,
            "Marks": "2/2" if is_correct else "0/2"
        })
    
    st.success(f"Multi-correct Score: {multi_score}/2")
    
    # Display total score
    st.markdown("---")
    percentage = (total_score / 15) * 100
    
    if percentage >= 80:
        grade = "A"
        emoji = "🌟"
        message = "Excellent work!"
    elif percentage >= 60:
        grade = "B"
        emoji = "👍"
        message = "Good job!"
    elif percentage >= 40:
        grade = "C"
        emoji = "📚"
        message = "Keep practicing!"
    else:
        grade = "D"
        emoji = "💪"
        message = "Need more practice!"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Score", f"{total_score}/15", f"{percentage:.1f}%")
    with col2:
        st.metric("Grade", grade, emoji)
    with col3:
        st.metric("Status", message)
    
    # Show detailed results
    st.markdown("### 📋 Detailed Results")
    df_results = pd.DataFrame(results_detail)
    st.dataframe(df_results, use_container_width=True)
    
    # Store results in session state
    st.session_state.quiz_score = total_score
    st.session_state.desc_scores = desc_scores
    st.session_state.quiz_percentage = percentage
    st.session_state.quiz_grade = grade
    st.session_state.quiz_attempts += 1
    # Email Results Button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📧 Send Results to My Email", use_container_width=True):
            # Prepare email body
            student = st.session_state.student_info
            email_body = f"""
            <html>
                <body style=\"font-family: Arial, sans-serif;\">
                    <h2>Quiz Results - AI Quiz Generator</h2>
                    <p>Dear <strong>{student['name']}</strong>,</p>
                    <p>Here are your quiz results:</p>
                    <table style=\"border-collapse: collapse; width: 100%;\">
                        <tr style=\"background-color: #f2f2f2;\">
                            <td style=\"padding: 8px; border: 1px solid #ddd;\"><strong>Student Name</strong></td>
                            <td style=\"padding: 8px; border: 1px solid #ddd;\">{student['name']}</td>
                        </tr>
                        <tr>
                            <td style=\"padding: 8px; border: 1px solid #ddd;\"><strong>Roll Number</strong></td>
                            <td style=\"padding: 8px; border: 1px solid #ddd;\">{student['roll_no']}</td>
                        </tr>
                        <tr style=\"background-color: #f2f2f2;\">
                            <td style=\"padding: 8px; border: 1px solid #ddd;\"><strong>Date & Time</strong></td>
                            <td style=\"padding: 8px; border: 1px solid #ddd;\">{student['timestamp']}</td>
                        </tr>
                    </table>
                    <h3>Score Breakdown:</h3>
                    <ul>
                        <li>MCQ Score: {mcq_score}/5</li>
                        <li>Descriptive Score: {sum(desc_scores)}/6 
                            <ul>
                                <li>Question 6: {desc_scores[0]}/3</li>
                                <li>Question 7: {desc_scores[1]}/3</li>
                            </ul>
                        </li>
                        <li>True/False Score: {tf_score}/2</li>
                        <li>Multi-correct MCQ Score: {multi_score}/2</li>
                    </ul>
                    <h3>Final Result:</h3>
                    <table style=\"border-collapse: collapse;\">
                        <tr>
                            <td style=\"padding: 10px; border: 2px solid #4CAF50; background-color: #f9f9f9;\">
                                <strong>Total Score:</strong> {total_score}/15 ({percentage:.1f}%)
                            </td>
                            <td style=\"padding: 10px; border: 2px solid #4CAF50; background-color: #f9f9f9;\">
                                <strong>Grade:</strong> {grade}
                            </td>
                        </tr>
                    </table>
                    <p style=\"margin-top: 20px;\"><em>{message}</em></p>
                    <hr>
                    <p style=\"font-size: 12px; color: #666;\">
                        This is an automated email from AI Quiz Generator.<br>
                        If you have any questions, please contact your teacher.
                    </p>
                </body>
            </html>
            """
            with st.spinner("Sending email..."):
                if send_gmail(
                    student['email'],
                    f"Quiz Results - {student['name']} ({student['roll_no']})",
                    email_body
                ):
                    st.success(f"✅ Results sent successfully to {student['email']}!")
                else:
                    st.error("❌ Failed to send email. Please check your email configuration.")
    # Option to retake quiz (only if attempts < 2)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        attempts_left = 2 - st.session_state.quiz_attempts
        if attempts_left > 0:
            if st.button("🔄 Take Another Quiz", use_container_width=True):
                # Reset quiz-related session state
                st.session_state.quiz_generated = False
                st.session_state.quiz_submitted = False
                st.session_state.quiz_data = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_score = None
                st.session_state.desc_scores = []
                st.session_state.quiz_start_time = None
                st.rerun()
        else:
            st.info("You have reached the maximum number of quiz attempts (2). No more attempts allowed.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>AI Quiz Generator v1.0 | Powered by Gemini & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)