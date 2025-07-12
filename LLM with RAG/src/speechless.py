
from transformers.pipelines import pipeline
import torch
import gradio as gr
import numpy as np
import whisper # or use your existing Whisper pipeline
import librosa
# Replace DiVA
pipe = pipeline(
    "text-generation",
    model="Menlo/Speechless-llama3.2-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
# Load Whisper for ASR
whisper_model = whisper.load_model("base")

def transcribe_whisper(sr, audio):
    if audio is None:
        return ""

    # Convert to mono if stereo
    y = audio.mean(axis=1) if audio.ndim > 1 else audio

    # Normalize safely
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    else:
        return ""

    # Resample to 16 kHz for Whisper
    y_resampled = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000)

    # Pad or trim to 30s as required by Whisper
    y_resampled = whisper.pad_or_trim(y_resampled)

    # Transcribe
    result = whisper_model.transcribe(y_resampled, fp16=False)
    return result["text"]

def speechless_chat(audio_input, prev_token=None, do_sample=False):
    sr, y = audio_input
    user_text = transcribe_whisper(sr, y)
    print(f"User text is {user_text}")
    prompt = f"User: {user_text}\nAssistant:"
    # if pipe:
    #     for out in pipe(prompt, max_new_tokens=256, do_sample=do_sample, stream=True):
    #         yield out["generated_text"]

demo = gr.Interface(
    fn=speechless_chat,
    inputs=gr.Audio(sources="microphone", type="numpy", streaming=False),
    outputs=gr.Textbox()
)
demo.launch()