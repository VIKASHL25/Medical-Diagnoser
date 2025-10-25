from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, pickle, numpy as np

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception as e:
    load_model = None
    pad_sequences = None

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, 'model')

app = FastAPI()
app.mount('/static', StaticFiles(directory=os.path.join(BASE_DIR, 'static')), name='static')
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))


class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = 17
        self.disease_labels = None
        self.prescription_labels = None
        self._load()
    def _load(self):
        try:
            model_path = os.path.join(MODEL_DIR, 'medical_diagnoser_model.h5')
            tok_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
            enc_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
            if load_model and os.path.exists(model_path):
                self.model = load_model(model_path)
            if os.path.exists(tok_path):
                with open(tok_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
            if os.path.exists(enc_path):
                with open(enc_path, 'rb') as f:
                    enc = pickle.load(f)
                    self.disease_labels = enc.get('disease')
                    self.prescription_labels = enc.get('prescription')
        except Exception as e:
            print('ModelService load error:', e)
    def predict(self, text):
        if self.model is None or self.tokenizer is None or pad_sequences is None:
            return {'disease':'Rheumatoid Arthritis (mock)','prescription':'DMARDs and NSAIDs (mock)','confidence':0.0}
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=self.max_length, padding='post')
        preds = self.model.predict(padded)
        try:
           
            if isinstance(preds, list) and len(preds) >= 2:
                disease_idx = int(np.argmax(preds[0], axis=1)[0])
                presc_idx = int(np.argmax(preds[1], axis=1)[0])
                conf = float(np.max(preds[0]))
            else:
                disease_idx = int(np.argmax(preds, axis=1)[0])
                presc_idx = None
                conf = float(np.max(preds))
        except Exception:
            disease_idx = int(np.argmax(preds, axis=1)[0])
            presc_idx = None
            conf = 0.0
        disease = self._decode(self.disease_labels, disease_idx) or str(disease_idx)
        presc = self._decode(self.prescription_labels, presc_idx) or (str(presc_idx) if presc_idx is not None else 'N/A')
        return {'disease':disease,'prescription':presc,'confidence':round(conf,3)}
    def _decode(self, labels, idx):
        if labels is None or idx is None:
            return None
        if isinstance(labels, (list,tuple)):
            return labels[idx] if 0 <= idx < len(labels) else None
        if isinstance(labels, dict):
            # assume inverse mapping
            return labels.get(idx) or None
        return None

service = ModelService()

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/predict', response_class=HTMLResponse)
async def predict(request: Request, patient_problem: str = Form(...)):
    result = service.predict(patient_problem)
    return templates.TemplateResponse('result.html', {'request': request, 'input': patient_problem, 'result': result})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)
