import gradio as gr
import numpy as np
from modeling_flow_mirror_bmodel import *

speaker_embedding = np.load("models/speaker_embedding.npz")['speaker_embedding_1']
hubert = CNHubert("models")
model = FlowmirrorForConditionalGeneration(model_dir="models", config=Config("configs/config.json"), device_id=0)

def layer_norm(x, epsilon=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    normalized_x = (x - mean) / (std + epsilon)
    return normalized_x

def app(question):
    ### preprocess audio ###
    sr, wav = question
    if wav.ndim == 2:
        print("avarage")
        wav = wav.mean(0)
    assert wav.ndim == 1, wav.ndim
    if wav.shape[0] < 50000:
        wav = np.pad(wav, (0, 50000 - wav.shape[0]))
    else:
        wav = wav[:50000]
    wav = layer_norm(wav)
    wav = np.reshape(wav,(1,-1))

    ### get input_ids by CNHubert ###
    input_ids = hubert.get_input_ids(wav)
    input_ids = np.pad(input_ids, ((0,0),(IDS_LENGTH - input_ids.shape[1], 0)))

    ### Flowmirror Generation ###
    generation, text_completion = model.generate(prompt_input_ids=input_ids, speaker_embedding=speaker_embedding)
    anwser = generation.squeeze()
    return anwser

    

# 创建 Gradio 界面
iface = gr.Interface(
    fn=app,
    inputs="audio",
    outputs="audio",
    title="心流知镜",
)

# 启动界面
iface.launch()
