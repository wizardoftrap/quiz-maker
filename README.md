# AI Quiz Generator for Students

This is a robust Streamlit app for student quizzes, powered by Gemini Flash 2.5 for question generation, Gemini embeddings for PDF notes, Pinecone for vector storage, and PyPDF for PDF processing.

## Features
- Teachers upload class notes (PDF) via command line.
- Students log in with their name, roll number, and email.
- Quiz is generated from teacher notes using Gemini.
- Quiz types: MCQ, descriptive, true/false, and multi-correct MCQ.
- Auto-grading and email results to students.
- 2 attempts per student (tracked per session/roll number).
- 20-minute timer per quiz attempt.
- Robust handling of Gemini's JSON output.

## Setup
1. **Clone the repository and navigate to the project folder.**
2. **Create a virtual environment and activate it:**
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Mac/Linux
   ```
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Set up your `.env` file:**
   - Add your Gemini, Pinecone, and SMTP credentials as environment variables.

5. **Upload teacher notes:**
   - Run the `upload_notes.py` script to upload and embed PDF notes.
      ```
      python upload_notes.py BasicsOfStatistics.pdf
      ```

6. **Run the app:**
   ```
   streamlit run app.py
   ```

## Usage
- Students fill in their details and start the quiz.
- Each student can attempt the quiz twice per session.
- Results are auto-graded and can be emailed.
- Teachers can update notes by re-running the upload script.

## Notes
- Quiz attempt tracking is per session.
- For persistent tracking, integrate a database and check attempts by roll number.

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies.

