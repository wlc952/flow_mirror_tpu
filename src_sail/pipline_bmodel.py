import numpy as np
import soundfile as sf
import time
from transformers import AutoFeatureExtractor
from modeling_flow_mirror_bmodel import *

SEQ_LENGTH = 512
IDS_LENGTH = 196


wav_data = "assets/question_example_3_MP3.wav"


hubert = CNHubert("models")
input_ids = hubert.get_input_ids(wav_data)
input_ids = np.pad(input_ids, ((0,0),(IDS_LENGTH - input_ids.shape[1], 0)))

### method 1
speaker_embedding = np.load("models/speaker_embedding.npz")['speaker_embedding_2']


# #### method 2
# import torch
# import torch.nn.functional as F
# feature_extractor = AutoFeatureExtractor.from_pretrained("configs")
# reference_audio_input = feature_extractor(sf.read(wav_data)[0],sampling_rate=16000, return_tensors="pt")
# audio_input = reference_audio_input['input_values']
# audio_input = F.pad(audio_input, (0, 50000 - audio_input.shape[2]))
# speaker_encoder = torch.jit.load("models/speaker_encoder.pt")
# speaker_embedding = speaker_encoder(audio_input).detach().numpy()


model = FlowmirrorForConditionalGeneration(model_dir="models", config=Config("configs/config.json"), device_id=0)

start = time.time()
i = 0
conti = True
while conti:
    generation, text_completion = model.generate(prompt_input_ids=input_ids, speaker_embedding=speaker_embedding)
    end = time.time()
    if (generation == 0.0).all(): conti = True
    else:  conti = False
    i += 1

print("Time taken: ", end - start)
print("i:", i-1)

audio_arr = generation.squeeze()
audio_arr = audio_arr - np.min(audio_arr)
max_audio=np.max(audio_arr)
if max_audio>1: audio_arr/=max_audio
audio_arr = (audio_arr * 32768).astype(np.int16)
sf.write("answer.wav", audio_arr, 16000)