from typing import Any, List, Tuple
import chromadb
import gradio as gr
from openai import OpenAI
from transformers.pipelines import pipeline
import numpy as np
from gradio import themes
import xxhash
from dataclasses import dataclass, field
import soundfile as sf

from lmstudio import load_paper_pdf, chunk_and_embed_document, persist_in_collection, generate_rag_response

openai_client = OpenAI(api_key="not-needed", base_url="http://localhost:1234/v1" )

chroma_client = chromadb.PersistentClient(path="./chroma_db/")

LOADER_STR = "♫♪ Loading... ♫♪"

@dataclass
class AppState:
    conversation : list = field(default_factory=list)
    stopped : bool = False
    model_outs: any = None

def process_audio(audio: tuple, state: AppState):
    print("Started Recording")
    print(f"Audio : {audio}, \nState: {state}")
    return audio, state


transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
iterator = 1
def response(state, input_audio):

    global iterator
    print("Finished Recording")
    
    print(f"input_audio : {input_audio}, \nState: {state}")
    dict = {
        "role": "assistant",
        "content": f"YOoo {iterator}",
    }
    iterator = iterator+1
    state.conversation.append(dict)
    return state, state.conversation

def transcribe_audio(state: AppState, input_audio: Tuple[int, np.ndarray]):
    
    file_name = f"/home/akhil/workspace/LLM-Tinkering/LLM with RAG/src/tmp/{xxhash.xxh32(bytes(input_audio[1])).hexdigest()}.wav"

    sf.write(file_name, input_audio[1], input_audio[0], format="wav")

    state.conversation.append(
        {"role": "user", "content": {"path": file_name, "mime_type": "audio/wav"}}
    )

    sample_rate, data = input_audio
    if data is None:
        yield state, state.conversation
        return

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    data /= np.max(np.abs(data))

    result = transcriber({"sampling_rate": sample_rate, "raw": data})["text"] #type: ignore
    state.conversation.append({
        "role": "user",
        "content": result,
    })
    print(f"Appended to conversation")
    state.conversation.append({"role": "assistant", "content": LOADER_STR})
    yield state, state.conversation  # display spinner immediately

    assistant_response = ""

    
    pages = load_paper_pdf()
    docs, embeddings = chunk_and_embed_document(pages)
    collection = persist_in_collection(docs=docs, embeddings=embeddings)

    for token in generate_rag_response(result, collection=collection, docs = docs):  # ← your streaming function
        assistant_response += token
        state.conversation[-1]["content"] = assistant_response
        yield state, state.conversation  # update in frontend

    print(f"[Assistant] {assistant_response}")

# def append_transcribed_audio_to_conversation(audio, state):
#     state, conversation = transcribe_audio(audio, state)


theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
    neutral_hue="stone",
)

js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded") ;
    var record = document.querySelector('.record-button');
    record.textContent = "Just Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        console.log("Speech started!");
        var record = document.querySelector('.record-button');
        if (record != null) {
          console.log(record);
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) {
          console.log(stop);
          stop.click();
        }
      }
    })
    myvad.start()
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () =>  {
    console.log("onnx loaded") 
    document.head.appendChild(script2)
  };
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

def start_recording_user(state: AppState):
    return None

with gr.Blocks(theme=theme, js=js) as demo:
    with gr.Row():
        input_audio = gr.Audio(
            label="Input Audio",
            sources=["microphone"],
            type="numpy",
            streaming=False,
            waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        )
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", type="messages")

    state = gr.State(value=AppState())
    print(type(input_audio))
    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )
    respond = input_audio.stop_recording(
        transcribe_audio, [state, input_audio], [state, chatbot]
    )
    restart = respond.then(start_recording_user, [state], [input_audio]).then(
        lambda state: state, state, state, js=js_reset
    )

    cancel = gr.Button("Restart Conversation", variant="stop")
    cancel.click(
        lambda: (AppState(), gr.Audio(recording=False)),
        None,
        [state, input_audio],
        cancels=[respond, restart],
    )

if __name__ == "__main__":
    demo.launch()