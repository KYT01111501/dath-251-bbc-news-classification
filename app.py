import gradio as gr
import joblib
import numpy as np
import os

print("[App] Working dir:", os.getcwd())

# ---- load artifacts with clear logs ----
try:
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    print("[App] Loaded tfidf_vectorizer.pkl")
    model = joblib.load("text_model.pkl")
    print("[App] Loaded text_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print("[App] Loaded label_encoder.pkl")
except Exception as e:
    print("[App] ERROR loading artifacts:", e)
    tfidf = model = label_encoder = None

def predict_text(text: str):
    if not text or not text.strip():
        return {}
    if tfidf is None or model is None:
        return {"Model not loaded": 1.0}
    X = tfidf.transform([text])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None and not isinstance(classes[0], str):
            classes = label_encoder.inverse_transform(classes)
        top_idx = np.argsort(proba)[::-1][:5]
        return {str(classes[i]): float(proba[i]) for i in top_idx}
    else:
        pred = model.predict(X)[0]
        if not isinstance(pred, str):
            pred = label_encoder.inverse_transform([pred])[0]
        return {str(pred): 1.0}

demo = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(lines=8, label="Paste article text here"),
    outputs=gr.Label(num_top_classes=5),
    title="BBC News Classifier (TF-IDF + ML)",
    description="Dán đoạn văn bản; app dự đoán: Business / Entertainment / Politics / Sport / Tech."
)

if __name__ == "__main__":
    # inbrowser=True tự mở trình duyệt; server_name=127.0.0.1 chạy local
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, share=False)
