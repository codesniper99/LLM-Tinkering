
from typing import Any
import gradio as gr
from dataclasses import dataclass, field





@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    stopped : bool = False
    model_outs : Any = None

def process_audio(audio: tuple, state: AppState):
    return audio, state

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
    console.log("vad loaded");
    var record = document.querySelector('.record-button');
    record.textContent = "Just Start Talking!"
    
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        var record = document.querySelector('.record-button');
        var player = document.querySelector('#streaming-out')
        if (record != null && (player == null || player.paused)) {
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) {
          stop.click();
        }
      }
    })
    myvad.start()
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

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


stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )
respond = input_audio.stop_recording(
        response, [state, input_audio], [state, chatbot]
    )


if __name__ == "__main__":
    demo.launch()

