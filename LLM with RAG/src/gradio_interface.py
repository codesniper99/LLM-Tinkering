# import gradio as gr
# import nemo.collections.asr as nemo_asr
# import numpy as np
# import torchaudio
# import torch
# # Load NeMo model (in WSL)
# asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# TARGET_SR = 16000

# def transcribe(stream, new_chunk):
#     sr, y = new_chunk
#     print(new_chunk)
#     if y is None:
#         return stream, ""
    
#     if y.ndim > 1:
#         y = y.mean(axis=1)
    
#     if sr != TARGET_SR:
#         y = torchaudio.functional.resample(
#             torch.tensor(y), orig_freq=sr, new_freq=TARGET_SR
#         ).numpy()
#         sr = TARGET_SR

#     # Normalize audio to [-1, 1]
#     y = y.astype(np.float32)
#     if np.max(np.abs(y)) > 0:
#         y /= np.max(np.abs(y))

#     # Append to stream buffer
#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y

#     # Transcribe current stream
#     text = asr_model.transcribe([stream])[0] #type: ignore
#     return stream, text

# # Build Gradio interface
# demo = gr.Interface(
#     fn=transcribe,
#     inputs=["state", gr.Audio(sources="microphone", streaming=True)],
#     outputs=["state", "text"],
#     live=True  # Set to True if you want reactivity (not required here)
# )

# demo.launch()  # use share=True to expose public URL if needed
import gradio as gr
from transformers.pipelines import pipeline
import numpy as np

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(stream, new_chunk):
    sr, y = new_chunk
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y
    return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]  

demo = gr.Interface(
    transcribe,
    ["state", gr.Audio(sources=["microphone"], streaming=True)],
    ["state", "text"],
    live=True,
)

demo.launch()