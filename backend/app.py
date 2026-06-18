"""
app.py  —  NeuraScan AI Flask server

"""

import os
import uuid
import sqlite3
import datetime
import traceback

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from dotenv import load_dotenv

load_dotenv()

import requests

# ── PATHS ───────────────────────────────────────────────────────────────────

THIS_FILE    = os.path.abspath(__file__)          # .../backend/app.py
BACKEND_DIR  = os.path.dirname(THIS_FILE)         # .../backend/
PROJECT_DIR  = os.path.dirname(BACKEND_DIR)       # .../BrainTumourDetector/

MODEL_DIR    = os.path.join(BACKEND_DIR, 'model')
TEMPLATE_DIR = os.path.join(PROJECT_DIR, 'templates')
STATIC_DIR   = os.path.join(PROJECT_DIR, 'static')
UPLOAD_DIR   = os.path.join(STATIC_DIR,  'uploads')
GRADCAM_DIR  = os.path.join(STATIC_DIR,  'gradcam')
REPORT_DIR   = os.path.join(STATIC_DIR,  'reports')
DB_PATH      = os.path.join(BACKEND_DIR, 'predictions.db')

# Create folders that must exist
for d in [UPLOAD_DIR, GRADCAM_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

print('\n=== NeuraScan AI starting ===')
print(f'BACKEND_DIR  : {BACKEND_DIR}')
print(f'PROJECT_DIR  : {PROJECT_DIR}')
print(f'TEMPLATE_DIR : {TEMPLATE_DIR}  [exists={os.path.isdir(TEMPLATE_DIR)}]')
print(f'STATIC_DIR   : {STATIC_DIR}  [exists={os.path.isdir(STATIC_DIR)}]')
print(f'MODEL_DIR    : {MODEL_DIR}  [exists={os.path.isdir(MODEL_DIR)}]')
print()

# ── FLASK ────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder = TEMPLATE_DIR,
    static_folder   = STATIC_DIR,
    static_url_path = '/static',
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16 MB
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-only-fallback-key-change-me')

# ── LOAD MODELS ──────────────────────────────────────────────────────────────
import predict as P
import gradcam as GC
import report  as RP

print('Loading models...')
try:
    P.load_models(MODEL_DIR)
    print('All models loaded OK\n')
except Exception as e:
    print(f'FATAL: could not load models: {e}\n')
    traceback.print_exc()
    raise SystemExit(
        f'\nNeuraScan AI cannot start: model loading failed ({e}).\n'
        f'Check that vgg16_final.keras / resnet50_final.keras exist in {MODEL_DIR}\n'
    )

# ── DATABASE ─────────────────────────────────────────────────────────────────
def init_db():
    # Delete corrupted DB if it exists
    if os.path.exists(DB_PATH):
        c = None
        try:
            c = sqlite3.connect(DB_PATH)
            c.execute('SELECT 1')
        except sqlite3.DatabaseError:
            print('DB corrupted — rebuilding')
            if c:
                try:
                    c.close()
                except Exception:
                    pass
            try:
                os.remove(DB_PATH)
            except OSError as e:
                raise SystemExit(f'NeuraScan AI cannot start: could not remove corrupted DB at {DB_PATH}: {e}')
        finally:
            if c:
                try:
                    c.close()
                except Exception:
                    pass

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT,
                patient_name TEXT,
                patient_id   TEXT,
                image_file   TEXT,
                prediction   TEXT,
                confidence   REAL,
                risk_level   TEXT,
                area_pct     REAL,
                report_file  TEXT
            )
        ''')
        conn.commit()
        print('DB ready')
    except sqlite3.Error as e:
        traceback.print_exc()
        raise SystemExit(f'NeuraScan AI cannot start: database setup failed: {e}')
    finally:
        try:
            conn.close()
        except Exception:
            pass

init_db()

def save_prediction(pname, pid, img_file, result, report_file):
    """
    Returns True on success, False on failure. Never raises — a failed
    history-log write should not prevent the user from seeing their result.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            'INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)',
            (None,
             datetime.datetime.now().isoformat(),
             pname, pid, img_file,
             result['pred_class'],
             result['confidence'],
             result['risk_level'],
             result['area_pct'],
             report_file)
        )
        conn.commit()
        return True
    except sqlite3.Error as e:
        print('save_prediction error:', e)
        traceback.print_exc()
        return False
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass



# ── ALLOWED FILE EXTENSIONS ──────────────────────────────────────────────────
ALLOWED = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}

def allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED
def local_medical_assistant(question):
    q = question.lower()

    responses = {
        "what is tumour":
        "A tumour is an abnormal growth of cells in the body. Tumours can be benign (non-cancerous) or malignant (cancerous).",

        "what is brain tumour":
        "A brain tumour is an abnormal growth of cells inside or around the brain. Some tumours are benign while others can be malignant.",

        "what is meningioma":
        "Meningioma is a tumour that develops from the meninges, the protective membranes surrounding the brain and spinal cord. Most meningiomas are benign and slow-growing.",

        "what is glioma":
        "Glioma is a tumour arising from glial cells in the brain. Gliomas can vary from low-grade to aggressive forms.",

        "what is pituitary tumour":
        "A pituitary tumour develops in the pituitary gland. Many pituitary tumours are non-cancerous and can affect hormone production.",

        "what is no tumour":
        "No Tumour means the AI model did not detect signs of the tumour classes it was trained to identify.",

        "what is low risk":
        "Low risk means the model has relatively low concern based on its prediction. Clinical evaluation is still recommended.",

        "what is medium risk":
        "Medium risk means the model has moderate confidence in its prediction and further medical review is advisable.",

        "what is high risk":
        "High risk means the model is highly confident in its prediction. Prompt consultation with a neurologist is recommended.",

        "how accurate is this system":
        "This AI system is an educational decision-support tool. Its accuracy depends on the training dataset and should not replace professional medical diagnosis.",

        "should i trust this result":
        "The result should be considered a preliminary AI assessment only. Always consult a qualified neurologist or radiologist for diagnosis.",

        "who made this system":
        "This Brain Tumour Detection System was developed as an AI-assisted medical imaging project using deep learning techniques."
    }

    for key, value in responses.items():
        if key in q:
            return value

    return None

# ── ROUTES ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', error=None)


@app.route('/predict', methods=['POST'])
def do_predict():
    # 1. Check file was uploaded
    if 'mri_file' not in request.files:
        return render_template('index.html', error='No file uploaded. Please select an MRI image.')

    f = request.files['mri_file']

    if f.filename == '':
        return render_template('index.html', error='No file selected.')

    if not allowed(f.filename):
        return render_template('index.html', error='Only JPG, PNG images are supported.')

    # 2. Read form data
    patient_name = (request.form.get('patient_name') or 'Anonymous').strip()
    patient_id   = (request.form.get('patient_id')   or 'PT-001').strip()

    # 3. Save uploaded image
    uid       = uuid.uuid4().hex[:12]
    ext       = f.filename.rsplit('.', 1)[1].lower()
    img_fname = f'{uid}.{ext}'
    img_path  = os.path.join(UPLOAD_DIR, img_fname)

    try:
        f.save(img_path)
    except Exception as e:
        print('Upload save error:', e)
        traceback.print_exc()
        return render_template('index.html', error='Could not save the uploaded file. Please try again.')

    try:
        # 4. Run prediction
        result = P.predict(img_path)

        # 5. Generate Grad-CAM
        gc = GC.generate_gradcam_overlay(img_path, GRADCAM_DIR)
        result['area_pct']         = gc['area_pct']
        result['heatmap_filename'] = gc['heatmap_filename']
        result['overlay_filename'] = gc['overlay_filename']

        # 6. Generate PDF
        ov_path = os.path.join(GRADCAM_DIR, gc['overlay_filename'])
        report_fname = RP.generate_report(
            orig_path    = img_path,
            overlay_path = ov_path,
            result       = result,
            save_dir     = REPORT_DIR,
            patient_name = patient_name,
            patient_id   = patient_id,
        )

        # 7. Save to DB (logged but non-fatal if it fails — the user still gets their result)
        if not save_prediction(patient_name, patient_id, img_fname, result, report_fname):
            print(f'Warning: prediction for {img_fname} was not logged to history DB')

    except Exception as e:
        print('Prediction pipeline error:', e)
        traceback.print_exc()

        # Clean up the orphaned upload since analysis never completed
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except OSError:
            pass

        return render_template(
            'index.html',
            error='Analysis failed while processing your scan. Please try a different image or try again shortly.'
        )

    # 8. Sort probs highest first for display
    sorted_probs = sorted(result['probs'].items(), key=lambda x: x[1], reverse=True)

    return render_template(
        'result.html',
        result       = result,
        patient_name = patient_name,
        patient_id   = patient_id,
        img_filename = img_fname,
        heatmap_file = result.get('heatmap_filename', ''),
        overlay_file = result.get('overlay_filename', ''),
        report_file  = report_fname,
        sorted_probs = sorted_probs,
        timestamp    = datetime.datetime.now().strftime('%d %b %Y, %H:%M'),
    )


@app.route('/download/<filename>')
def download(filename):
    safe_name = secure_filename(filename)
    if not safe_name or safe_name != filename:
        return jsonify({'error': 'Invalid filename.'}), 400

    file_path = os.path.join(REPORT_DIR, safe_name)
    if not os.path.isfile(file_path):
        return jsonify({'error': 'Report not found. It may have expired or the analysis failed.'}), 404

    try:
        return send_from_directory(REPORT_DIR, safe_name, as_attachment=True)
    except Exception as e:
        print('Download error:', e)
        traceback.print_exc()
        return jsonify({'error': 'Could not download the report. Please try again.'}), 500


@app.route('/history')
def history():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT * FROM predictions ORDER BY id DESC LIMIT 20'
        ).fetchall()
        return jsonify([dict(r) for r in rows])
    except sqlite3.Error as e:
        print('History query error:', e)
        traceback.print_exc()
        return jsonify({'error': 'Could not load history right now. Please try again.'}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json(force=True) or {}
    question = (data.get('question') or '').strip()
    context = data.get('context', {})

    if not question:
        return jsonify({'reply': 'Please type a question.'})

    # Offline FAQ Assistant
    offline_reply = local_medical_assistant(question)

    if offline_reply:
        return jsonify({'reply': offline_reply})

    try:
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise RuntimeError("SARVAM_API_KEY is not set in the environment")

        headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json"
        }

        system = (
            f"You are a compassionate AI medical assistant explaining a brain MRI result. "
            f"The AI found: {context.get('pred_class', 'unknown').upper()} with "
            f"{context.get('confidence', 0):.1f}% confidence. "
            f"Risk level: {context.get('risk', 'Unknown')}. "
            f"Explain in plain, simple language. "
            f"Always advise consulting a qualified neurologist. "
            f"Never make definitive clinical claims."
        )

        payload = {
            "model": "sarvam-30b",
            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "temperature": 0.2
        }

        response = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()  # turn 4xx/5xx into a catchable exception

        data = response.json()
        reply = data["choices"][0]["message"]["content"]

        return jsonify({"reply": reply})

    except requests.exceptions.Timeout:
        print("Chatbot error: request to Sarvam API timed out")
        return jsonify({
            "reply": "The assistant is taking too long to respond. Please try again in a moment."
        })

    except requests.exceptions.ConnectionError as e:
        print("Chatbot error: connection failed —", str(e))
        return jsonify({
            "reply": "I couldn't reach the AI assistant service right now. Please check your connection and try again."
        })

    except requests.exceptions.HTTPError as e:
        print("Chatbot error: bad HTTP response —", str(e))
        traceback.print_exc()
        return jsonify({
            "reply": "The AI assistant service returned an error. Please try again shortly, or rephrase your question."
        })

    except (KeyError, IndexError, ValueError) as e:
        print("Chatbot error: unexpected response format —", str(e))
        traceback.print_exc()
        return jsonify({
            "reply": "I received an unexpected response from the assistant service. Please try again."
        })

    except Exception as e:
        print("Chatbot error:", str(e))
        traceback.print_exc()
        return jsonify({
            "reply": "Something went wrong while processing your question. Please try again, "
                     "and consult a qualified neurologist for any medical concerns."
        })
# ── APP-WIDE ERROR HANDLERS ──────────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    if request.path == '/predict':
        return render_template(
            'index.html',
            error='That file is too large. Please upload an image under 16 MB.'
        ), 413
    return jsonify({'error': 'Uploaded file is too large (max 16 MB).'}), 413


@app.errorhandler(404)
def not_found(e):
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({'error': 'Not found.'}), 404
    return render_template('index.html', error='Page not found.'), 404


@app.errorhandler(500)
def server_error(e):
    traceback.print_exc()
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({'error': 'Internal server error. Please try again.'}), 500
    return render_template(
        'index.html',
        error='Something went wrong on our end. Please try again shortly.'
    ), 500


# ── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    print(f'Starting Flask on http://0.0.0.0:{port}  (debug={debug_mode})')
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
