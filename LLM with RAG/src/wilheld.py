import io
import os
import time
import traceback
from dataclasses import dataclass, field

import gradio as gr
import librosa
import numpy as np
import pvorca
import soundfile as sf
import spaces
import torch
import xxhash
from datasets import Audio
from transformers import AutoModel
from transformers.modeling_outputs import CausalLMOutputWithPast

orca = pvorca.create(
    access_key=os.getenv("ORCA_KEY"),
    model_path="./static/orca_params_masculine.pv",
)
LOADER_STR = "♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪loading♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪loading♫♪.ılılıll|̲̅̅●̲̅̅|̲̅̅=̲̅̅|̲̅̅●̲̅̅|llılılı.♫♪♫"
if gr.NO_RELOAD:
    diva_model = AutoModel.from_pretrained(
        "WillHeld/DiVA-llama-3-v0-8b", trust_remote_code=True
    )

    resampler = Audio(sampling_rate=16_000)


@spaces.GPU
@torch.no_grad
def diva_audio(audio_input, do_sample=False, temperature=0.001, prev_outs=None):
    sr, y = audio_input
    x = xxhash.xxh32(bytes(y)).hexdigest()
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    a = resampler.decode_example(
        resampler.encode_example({"array": y, "sampling_rate": sr})
    )
    yield from diva_model.generate_stream(
        a["array"],
        (
            "Your name is DiVA, which stands for Distilled Voice Assistant. You were trained with early-fusion training to merge OpenAI's Whisper and Meta AI's Llama 3 8B to provide end-to-end voice processing. You should respond in a conversational style. The user is talking to you with their voice and you are responding with text. Use fewer than 20 words."
            if prev_outs == None
            else None
        ),
        do_sample=do_sample,
        max_new_tokens=256,
        init_outputs=prev_outs,
        return_outputs=True,
    )


@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    stopped: bool = False
    model_outs: any = None


def process_audio(audio: tuple, state: AppState):
    return audio, state


@spaces.GPU(duration=40, progress=gr.Progress(track_tqdm=True))
def response(state: AppState, audio: tuple):
    if not audio:
        return AppState()

    file_name = f"/tmp/{xxhash.xxh32(bytes(audio[1])).hexdigest()}.wav"

    sf.write(file_name, audio[1], audio[0], format="wav")

    state.conversation.append(
        {"role": "user", "content": {"path": file_name, "mime_type": "audio/wav"}}
    )
    if state.model_outs is None:
        gr.Warning(
            "The first response might take a second to generate as DiVA is loaded from Disk to the ZeroGPU!"
        )
    state.conversation.append(
        {
            "role": "assistant",
            "content": LOADER_STR,
        }
    )
    yield state, state.conversation, None
    if spaces.config.Config.zero_gpu:
        if state.model_outs is not None:
            state.model_outs = tuple(
                tuple(torch.tensor(vec).cuda() for vec in tup)
                for tup in state.model_outs
            )
        causal_outs = (
            CausalLMOutputWithPast(past_key_values=state.model_outs)
            if state.model_outs
            else None
        )
    else:
        causal_outs = state.model_outs
    state.model_outs = None
    prev_outs = causal_outs
    stream = orca.stream_open()

    buff = []
    for resp, outs in diva_audio(
        (audio[0], audio[1]),
        prev_outs=(prev_outs if prev_outs is not None else None),
    ):
        prev_resp = state.conversation[-1]["content"]
        if prev_resp == LOADER_STR:
            prev_resp = ""
        state.conversation[-1]["content"] = resp
        pcm = stream.synthesize(resp[len(prev_resp) :])
        audio_chunk = None
        if pcm is not None:
            buff.extend(pcm)
        if len(buff) > (orca.sample_rate * 2):
            mp3_io = io.BytesIO()
            sf.write(
                mp3_io,
                np.asarray(buff[: orca.sample_rate]).astype(np.int16),
                orca.sample_rate,
                format="mp3",
            )
            audio_chunk = mp3_io.getvalue()
            mp3_io.close()
            buff = buff[orca.sample_rate :]
        yield state, state.conversation, audio_chunk

    del outs.logits
    del outs.hidden_states
    if spaces.config.Config.zero_gpu:
        outs = tuple(
            tuple(vec.cpu().numpy() for vec in tup) for tup in outs.past_key_values
        )
    audio_chunk = None
    pcm = stream.flush()
    if pcm is not None:
        mp3_io = io.BytesIO()
        sf.write(
            mp3_io,
            np.asarray(buff + pcm).astype(np.int16),
            orca.sample_rate,
            format="mp3",
        )
        audio_chunk = mp3_io.getvalue()
        mp3_io.close()
    stream.close()
    yield (
        AppState(conversation=state.conversation, model_outs=outs),
        state.conversation,
        audio_chunk,
    )


def start_recording_user(state: AppState):
    return None


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
        var record = document.querySelector('.record-button');
        var player = document.getElementById("streaming_out").querySelector(".standard-player")
        if (record != null && (player == null || player.paused)) {
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
    with gr.Row(max_height="50vh"):
        output_audio = gr.Audio(
            label="Output Audio",
            streaming=True,
            autoplay=True,
            visible=True,
            elem_id="streaming_out",
        )
    state = gr.State(value=AppState())
    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )
    respond = input_audio.stop_recording(
        response, [state, input_audio], [state, chatbot, output_audio]
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

