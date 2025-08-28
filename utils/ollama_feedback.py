"""
Ollama utilities:
- Get natural conversational English feedback
"""

import ollama

def ollama_feedback(transcript, model_name):
    """
    Get natural, conversational feedback for spoken English from Ollama.
    Args:
        transcript (str): Transcribed user speech
        model_name (str): Ollama model name
    Returns:
        str: Feedback text
    """
    client = ollama.Client()
    messages = [
        {"role": "system", "content": 
         "You are an English speaking teacher giving feedback in a friendly, natural way. "
         "Respond as if you were talking to the student in person. "
         "Get a short, natural feedback with improvement suggestions for spoken English."
         "Keep the entire response concise."},
        {"role": "user", "content": transcript}
    ]
    try:
        response = client.chat(model=model_name, messages=messages)
        feedback_text = response.message["content"]
        print("💬 Ollama Response:\n", feedback_text)
        return feedback_text
    except Exception as e:
        print(f"❌ Ollama call failed: {e}")
        return "Failed to get feedback"
