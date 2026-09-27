"""
Unit tests for ChatbotEngine (Conversational Medical Co-Pilot).
File: tests/test_chatbot_engine.py
"""

import pytest
from PIL import Image
import numpy as np

from app.chatbot_engine import ChatbotEngine, ChatResponse
from app.inference_service import PredictionResult


@pytest.fixture
def dummy_prediction():
    heatmap = np.zeros((7, 7), dtype=np.float32)
    overlay = Image.new("RGB", (224, 224), color="blue")
    return PredictionResult(
        class_name="Malignant",
        class_idx=1,
        confidence=0.985,
        probabilities={"Benign": 0.01, "Malignant": 0.985, "Normal": 0.005},
        heatmap=heatmap,
        overlay_image=overlay,
        latency_ms=18.4
    )


def test_chatbot_text_greeting():
    engine = ChatbotEngine()
    response = engine.respond(user_text="Halo, apa itu PulmoScan AI?")
    assert isinstance(response, ChatResponse)
    assert len(response.text) > 20
    assert any(w in response.text.lower() for w in ["pulmoscan", "ct", "skrining", "kanker paru"])


def test_chatbot_format_diagnosis(dummy_prediction):
    engine = ChatbotEngine()
    response = engine.format_diagnostic_report(dummy_prediction, patient_id="TCGA-01")
    assert isinstance(response, ChatResponse)
    assert "Malignant" in response.text or "Kanker Ganas" in response.text
    assert "98.5%" in response.text or "98.50%" in response.text
    assert "Fleischner" in response.text
    assert response.has_image is True


def test_chatbot_follow_up_question(dummy_prediction):
    engine = ChatbotEngine()
    # Set context
    engine.last_prediction = dummy_prediction
    response = engine.respond(user_text="Kenapa didiagnosis malignant dan apa tindakan selanjutnya?")
    assert isinstance(response, ChatResponse)
    assert any(term in response.text.lower() for term in ["biopsi", "pet-ct", "onkologi", "ganas", "spikulasi"])
