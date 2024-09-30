import json
from tpu_perf.infer import SGInfer
import numpy as np
import os
import time
import onnxruntime as ort
from transformers import AutoTokenizer, AutoFeatureExtractor
import soundfile as sf
from modeling_flow_mirror_bmodel import *
SEQ_LENGTH = 512
IDS_LENGTH = 196

class BmodelLoader:
    def __init__(self, model_path="", batch=1, device_id=0) :
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        
    def __call__(self, args):
        start = time.time()
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        return results



wav_data = "/workspace/flow_mirror/model_weights/assets/question_example_3_MP3.mp3"


hubert = CNHubert("/workspace/flow_mirror/flow_mirror_s/onnx/hubert", device_id=2)
input_ids = hubert.get_input_ids(wav_data)
input_ids = np.pad(input_ids, ((0,0),(IDS_LENGTH - input_ids.shape[1], 0)))

speaker_embedding = np.load("/workspace/flow_mirror/model_weights/speaker_embedding.npz")['speaker_embedding_1']
print(speaker_embedding.dtype)

# feature_extractor = AutoFeatureExtractor.from_pretrained("/workspace/flow_mirror/flow_mirror_s/hubert_kmeans")
# reference_audio_input = feature_extractor(sf.read(wav_data)[0],sampling_rate=16000, return_tensors="pt")
# audio_input = reference_audio_input['input_values']
# audio_input = F.pad(audio_input, (0, 50000 - audio_input.shape[2]))

# speaker_encoder = torch.jit.load("/workspace/flow_mirror/flow_mirror_s/onnx/speaker_encoder.pt")
# speaker_embedding = speaker_encoder(audio_input).detach().numpy()
# print(speaker_embedding.dtype)


model = FlowmirrorForConditionalGeneration(model_dir="/workspace/flow_mirror/flow_mirror_s/onnx", config=Config("/workspace/flow_mirror/weights/config.json"),device_id=11)
generation, text_completion = model.generate(prompt_input_ids=input_ids, speaker_embedding=speaker_embedding)
if not (generation == 0.0).all() :
    max_value = np.max(np.abs(generation))
    # 将数据缩放到 [-1, 1]
    if max_value != 0:  # 避免除以零
        normalized_data = generation / max_value
    print(normalized_data.shape)
    normalized_data = normalized_data.squeeze()
    sf.write("answer.wav", normalized_data, 16000)
