import copy
import json
import inspect
import soundfile as sf
import numpy as np
import time
from transformers import AutoTokenizer
import onnxruntime as ort
from tpu_perf.infer import SGInfer
from itertools import groupby
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

SEQ_LENGTH = 512
IDS_LENGTH = 196

class BmodelLoader:
    def __init__(self, model_path="", batch=1, device_id=0):
        self.model = SGInfer(model_path, batch=batch, devices=[device_id])
        
    def __call__(self, args):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        return results


#####################   Tools of functions   #####################
def apply_delay_pattern_mask(input_ids, decoder_pad_token_mask):
    seq_len = input_ids.shape[-1]
    decoder_pad_token_mask = decoder_pad_token_mask[..., :seq_len]
    input_ids = np.where(decoder_pad_token_mask == -1, input_ids, decoder_pad_token_mask)
    return input_ids

def build_delay_pattern_mask(input_ids, bos_token_id=1025, pad_token_id=1024, max_length=513, num_codebooks=9):
    """
    Build a delayed pattern mask to the input_ids using numpy. Each codebook is offset by the previous codebook by one.
    """
    # Reshape input_ids: (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
    input_ids = input_ids.reshape(-1, num_codebooks, input_ids.shape[-1])
    bsz, num_codebooks, seq_len = input_ids.shape

    input_ids_shifted = np.ones((bsz, num_codebooks, max_length), dtype=np.int32) * -1

    # Apply the mask only if the sequence length is large enough
    if max_length < 2 * num_codebooks - 1:
        return input_ids.reshape(bsz * num_codebooks, -1), input_ids_shifted.reshape(bsz * num_codebooks, -1)

    # Fill the shifted ids with the prompt entries, offset by the codebook index
    for codebook in range(num_codebooks):
        input_ids_shifted[:, codebook, codebook:seq_len + codebook] = input_ids[:, codebook]

    # Construct a pattern mask that indicates the positions of padding tokens for each codebook
    # Upper triangular part (the EOS padding)
    eos_delay_pattern = np.triu(np.ones((num_codebooks, max_length), dtype=bool), k=max_length - num_codebooks + 1)
    # Lower triangular part (the BOS padding)
    bos_delay_pattern = np.tril(np.ones((num_codebooks, max_length), dtype=bool))

    bos_mask = ~bos_delay_pattern
    eos_mask = ~eos_delay_pattern
    mask = ~(bos_delay_pattern + eos_delay_pattern)

    input_ids_shifted = np.where(mask, input_ids_shifted, -1)
    input_ids_shifted = np.where(~bos_mask, bos_token_id, input_ids_shifted)
    input_ids_shifted = np.where(~eos_mask, pad_token_id, input_ids_shifted)

    # Find the first position to start generating (first -1 token in first codebook)
    first_codebook_ids = input_ids_shifted[:, 0, :]
    start_ids = np.where(first_codebook_ids == -1)[1]
    
    if len(start_ids) > 0:
        first_start_id = min(start_ids)
    else:
        # No tokens that need to be filled
        first_start_id = seq_len

    # (bsz * num_codebooks, seq_len) -> (bsz, num_codebooks, seq_len)
    pattern_mask = input_ids_shifted.reshape(bsz * num_codebooks, -1)
    input_ids_shifted = input_ids_shifted[..., :first_start_id].reshape(bsz * num_codebooks, -1)
    
    return input_ids_shifted, pattern_mask

def prepare_4d_causal_attention_mask(input_shape):
    attn_mask_converter = AttentionMaskConverter(is_causal=True)
    key_value_length = input_shape[-1]
    attention_mask = attn_mask_converter.to_causal_4d(input_shape[0], input_shape[-1], key_value_length)
    return attention_mask

def pad_sequence(sequences, padding_value=0):
    # 首先计算最长序列的长度
    max_length = max(len(seq) for seq in sequences)
    # 创建一个填充后的数组，形状为 (batch_size, max_length)
    padded_array = np.full((len(sequences), max_length), padding_value)
    # 将每个序列填充到相应的行
    for i, seq in enumerate(sequences):
        padded_array[i, :len(seq)] = seq
    return padded_array


#####################   Tools of classes   #####################
class LogitsProcessorList(list):
    def __call__(self, scores: np.ndarray) -> np.ndarray:
        for processor in self:
            scores = processor(scores)
        return scores
    
class TemperatureLogitsWarper:
    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = f"`temperature` (={temperature}) 必须为正浮点数，否则 logits 分数将无效。"
            raise ValueError(except_msg)
        self.temperature = temperature

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        return scores / self.temperature

class TopKLogitsWarper:
    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")
        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        top_k = min(self.top_k, scores.shape[-1])
        kth_values = np.partition(scores, -top_k, axis=-1)[:, -top_k]
        indices_to_remove = scores < np.expand_dims(kth_values, axis=-1)
        scores_processed = np.where(indices_to_remove, self.filter_value, scores)
        return scores_processed

class StoppingCriteriaList(list):
    def __call__(self, input_ids: np.ndarray, scores: np.ndarray, **kwargs) -> np.ndarray:
        is_done = np.full((input_ids.shape[0],), False, dtype=bool)
        for criteria in self:
            is_done = np.logical_or(is_done, criteria(input_ids, scores, **kwargs))
        return is_done

class EosTokenCriteria:
    def __init__(self, eos_token_id: Union[int, List[int], np.ndarray]):
        if not isinstance(eos_token_id, np.ndarray):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = np.array(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray, **kwargs) -> np.ndarray:
        eos_token_id = self.eos_token_id
        last_token = input_ids[:, -1]  # Get the last token in each sequence
        # Check if any of the last tokens in the sequences match any eos_token_id
        is_done = np.isin(last_token, eos_token_id)
        return is_done

class MaxLengthCriteria:
    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray, **kwargs) -> np.ndarray:
        return np.array([input_ids.shape[1] >= self.max_length])

class StopStringCriteria:
    def __init__(self, stop_strings: List[str], tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray, **kwargs) -> np.ndarray:
        decoded_sequences = [self.tokenizer.decode(seq) for seq in input_ids]
        is_done = np.array([any(stop_str in seq for stop_str in self.stop_strings) for seq in decoded_sequences])
        return is_done

class AttentionMaskConverter:
    def __init__(self, is_causal: bool):
        self.is_causal = is_causal

    def to_causal_4d(self, batch_size, query_length, key_value_length):
        if not self.is_causal:
            raise ValueError("Please use `to_causal_4d` only if `is_causal` is set to True.")
        # Create a causal mask
        causal_mask = self._make_causal_mask(query_length, key_value_length)
        # Expand mask to 4D (batch_size, 1, query_length, key_value_length)
        expanded_mask = np.expand_dims(causal_mask, axis=(0, 1)).astype(np.float32)
        return expanded_mask

    def _make_causal_mask(self, query_length, key_value_length):
        mask = np.full((query_length, key_value_length), -10000.0, dtype=np.float32)
        if self.is_causal:
            mask[np.tril_indices(query_length, k=0)] = 0.0
        return mask

class Config:
    def __init__(self, config_data):
        # 允许传入文件路径或者直接传递字典数据
        if isinstance(config_data, str):
            with open(config_data, 'r') as file:
                self._config_data = json.load(file)
        elif isinstance(config_data, dict):
            self._config_data = config_data
        else:
            raise ValueError("config_data should be either a file path or a dictionary")

    def __getattr__(self, name):
        # 支持递归嵌套属性访问
        value = self._config_data.get(name, None)

        if isinstance(value, dict):
            return Config(value)  # 递归返回Config对象
        return value

    def __getitem__(self, key):
        # 支持通过字典方式访问
        return self.__getattr__(key)

    def update(self, new_data):
        # 递归更新嵌套字典
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    recursive_update(d[k], v)
                else:
                    d[k] = v
        recursive_update(self._config_data, new_data)

    def as_dict(self):
        # 返回字典形式
        return self._config_data

    def __repr__(self):
        # 直接返回字典的表示
        return repr(self._config_data)



#####################   HubertModel   #####################
class CNHubert:
    def __init__(self, model_dir, batch=1, device_id=0):
        self.hubert = BmodelLoader(model_dir + "/hubert.bmodel", batch=batch, device_id=device_id)
        self.wav2vec2 = BmodelLoader(model_dir + "/wav2vec2.bmodel", batch=batch, device_id=device_id)
        self.kmeans = BmodelLoader(model_dir + "/k_means.bmodel", batch=batch, device_id=device_id)

    def layer_norm(self, x, epsilon=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        normalized_x = (x - mean) / (std + epsilon)
        return normalized_x

    def read_audio(self, path):
        wav, sr = sf.read(path)
        if wav.ndim == 2:
            print("avarage")
            wav = wav.mean(0)
        assert wav.ndim == 1, wav.ndim
        wav = np.pad(wav, (0, max(0, 50000 - wav.shape[0])))[:50000]
        wav = self.layer_norm(wav)
        wav = np.reshape(wav,(1,-1))
        return wav
    
    def deduplicates(self, cluster_ids):
        return [key for key, _ in groupby(cluster_ids)]
    
    def convert_label_to_text(self, label):
        text = ""
        for i in label:
            text += f"<|audio_{i}|>"
        return text

    def __call__(self, source: np.array):
        feats = self.hubert([source])[0]
        feats = self.wav2vec2([feats])[0].squeeze(0)
        label = self.kmeans([feats])[0]
        return label
    
    def get_input_ids(self, wav_data):
        wav = self.read_audio(wav_data).astype(np.float32)
        codes = self(wav)
        codes = self.deduplicates(codes)
        label_text = self.convert_label_to_text(codes)
        prompt = f"<|spk_embed|><|startofaudio|>{label_text}<|endofaudio|><|startofcont|>"
        tokenizer = AutoTokenizer.from_pretrained("/workspace/flow_mirror/model_weights/tokenizer")
        input_ids = tokenizer(prompt, return_tensors="np").input_ids
        return input_ids.astype(np.int32)


#####################   Flowmirror   #####################
class FlowmirrorModel:
    def __init__(self, model_dir, config=None, batch=1, device_id=4):
        self.block = []
        self.block_cache = []
        if config is not None:
            self.num_codebooks = config.num_codebooks
            self.num_blocks = config.num_hidden_layers
        else:
            self.num_codebooks = 9
            self.num_blocks = 28
        for i in range(self.num_blocks):
            self.block.append(BmodelLoader(f"{model_dir}/block_{i}.bmodel", batch=batch, device_id=device_id))
            self.block_cache.append(BmodelLoader(f"{model_dir}/block_cache_{i}.bmodel", batch=batch, device_id=device_id))
        self.inputs_embeds = BmodelLoader(f"{model_dir}/inputs_embeds.bmodel", batch=batch, device_id=device_id)
        self.layer_norm = BmodelLoader(f"{model_dir}/layer_norm.bmodel", batch=batch, device_id=device_id)
        self.layer_norm_cache = BmodelLoader(f"{model_dir}/layer_norm_cache.bmodel", batch=batch, device_id=device_id)

    def __call__(self, prompt_hidden_states = None, k_cache_list = None, v_cache_list = None, cache_lenth = None, input_ids = None):
        if input_ids is not None:
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        else:
            input = None

        inputs_embeds = None

        if input is not None:
            inputs_embeds = self.inputs_embeds([input])[0]

        if prompt_hidden_states is not None:
            inputs_embeds = prompt_hidden_states

        _, seq_len, _ = inputs_embeds.shape

        if k_cache_list is not None:
            position_ids = np.array([[cache_lenth - 1]], dtype=np.int32)
        else:
            position_ids = np.expand_dims(np.arange(0, IDS_LENGTH, dtype=np.int32), 0)

        input_shape = inputs_embeds.shape[:-1]
        hidden_states = inputs_embeds

        if k_cache_list is None:
            k_cache_list = [None] * self.num_blocks
            v_cache_list = [None] * self.num_blocks
            attention_mask = prepare_4d_causal_attention_mask(input_shape)
            zero_pad = np.zeros((1, 2, SEQ_LENGTH - IDS_LENGTH, 128), dtype=np.float32)
        else:
            attention_mask = np.zeros((1, 1, 1, SEQ_LENGTH + 1), dtype=np.float32)
            attention_mask[:, :, :, cache_lenth:SEQ_LENGTH] = -10000.0

        for idx in range(self.num_blocks):
            if k_cache_list[idx] is not None:
                input_list = [hidden_states, position_ids, attention_mask, k_cache_list[idx], v_cache_list[idx]]
                hidden_states, ik, iv  = self.block_cache[idx](input_list)
                k_cache_list[idx][:,:,cache_lenth:cache_lenth+1] = ik
                v_cache_list[idx][:,:,cache_lenth:cache_lenth+1] = iv
            else:
                hidden_states, ik, iv  = self.block[idx]([hidden_states, position_ids, attention_mask])
                ik = np.concatenate([ik, zero_pad], axis=2)
                iv = np.concatenate([iv, zero_pad], axis=2)
                k_cache_list[idx] = ik
                v_cache_list[idx] = iv

        if cache_lenth is not None:
            cache_lenth += 1
            hidden_states = self.layer_norm_cache([hidden_states])[0]
        else:
            cache_lenth = IDS_LENGTH
            hidden_states = self.layer_norm([hidden_states])[0]
        
        return hidden_states, k_cache_list, v_cache_list, cache_lenth

class FlowmirrorForCausalLM:
    def __init__(self, model_dir, config=None, batch=1, device_id=4):
        self.model = FlowmirrorModel(model_dir, config)
        self.text_lm_head = BmodelLoader(model_dir + "/text_lm_head.bmodel", batch=batch, device_id=device_id)
        self.text_lm_head_cache = BmodelLoader(model_dir + "/text_lm_head_cache.bmodel", batch=batch, device_id=device_id)
        self.audio_lm_head = BmodelLoader(model_dir + "/audio_lm_heads.bmodel", batch=batch, device_id=device_id)
        self.num_codebooks = config.num_codebooks if config is not None else 9

    def __call__(self, prompt_hidden_states=None, k_cache_list=None, v_cache_list=None, cache_lenth=None, input_ids=None, modality=None):
        hidden_states, new_k_cache_list, new_v_cache_list, cache_lenth = self.model(
            prompt_hidden_states=prompt_hidden_states,
            k_cache_list=k_cache_list,
            v_cache_list=v_cache_list,
            cache_lenth=cache_lenth,
            input_ids = input_ids,
        )
        if modality == "audio":
            lm_logits = self.audio_lm_head([hidden_states])[0]
        else:
            if hidden_states.shape[-2] == 1:
                lm_logits = self.text_lm_head_cache([hidden_states])[0]
            else:
                lm_logits = self.text_lm_head([hidden_states])[0]
        return lm_logits, new_k_cache_list, new_v_cache_list, cache_lenth
    
    def build_delay_pattern_mask(self, input_ids, bos_token_id, pad_token_id, max_length):
        return build_delay_pattern_mask(input_ids, bos_token_id, pad_token_id, max_length, self.num_codebooks)

    def apply_delay_pattern_mask(self, input_ids, decoder_pad_token_mask):
        return apply_delay_pattern_mask(input_ids, decoder_pad_token_mask)
    

#####################   Generation   #####################
class FlowmirrorForConditionalGeneration:
    def __init__(self, model_dir, config=None, batch=1, device_id=4):
        self.decoder = FlowmirrorForCausalLM(model_dir + "/flowmirrormodel", config.decoder if config else None, batch, device_id)
        # self.audio_encoder = ort.InferenceSession(model_dir + "/prepare/audio_encoder_dynamic.onnx")
        self.audio_encoder = BmodelLoader(model_dir + "/prepare/audio_encoder.bmodel", batch=batch, device_id=device_id)

        self.embed_prompts = BmodelLoader(model_dir + "/prepare/embed_prompts.bmodel", batch=batch, device_id=device_id)
        self.embed_prompts_cache = BmodelLoader(model_dir + "/prepare/embed_prompts_cache.bmodel", batch=batch, device_id=device_id)
        self.enc_to_dec_proj = BmodelLoader(model_dir + "/prepare/enc_to_dec_proj.bmodel", batch=batch, device_id=device_id)
        self.sample_output = ort.InferenceSession(model_dir + "/prepare/sample_output.onnx")
        self.config = config
        self.generation_config = Config({
                                            "bos_token_id": 1025,
                                            "decoder_start_token_id": 1025,
                                            "do_sample": True,
                                            "eos_token_id": 1024,
                                            "max_length": 513,
                                            "max_new_tokens": 512,
                                            "min_new_tokens": 10,
                                            "pad_token_id": 1024,
                                            "temperature": 0.9,
                                            "top_k": 50,
                                            })

    def __call__(self, prompt_hidden_states=None, k_cache_list=None, v_cache_list=None, cache_lenth=None, decoder_input_ids=None, speaker_embedding=None, modality=None, **kwargs,):
        # No.0 处理
        if prompt_hidden_states is not None and k_cache_list is None:
            speaker_embedding = self.enc_to_dec_proj([speaker_embedding])[0]
            prompt_hidden_states[:, 0, :] = speaker_embedding

        # Decode
        lm_logits, k_cache_list, v_cache_list, cache_lenth = self.decoder(
            prompt_hidden_states=prompt_hidden_states,
            k_cache_list=k_cache_list,
            v_cache_list=v_cache_list,
            cache_lenth=cache_lenth,
            input_ids=decoder_input_ids,
            modality=modality,
        )
        return lm_logits, k_cache_list, v_cache_list, cache_lenth

    def _prepare_model_inputs(self, inputs = None, bos_token_id = None,  model_kwargs = None):
        # 1. retrieve all kwargs that are non-None or non-model input related.
        input_name = 'input_ids'
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}
        # 2. check whether model_input_name is passed as kwarg
        # 3. In the presence of `inputs_embeds` for text models:
        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _maybe_initialize_input_ids_for_generation(self, inputs = None, bos_token_id = None, model_kwargs = None):
        if inputs is not None:
            return inputs
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, np.ndarray):
                batch_size = value.shape[0]
                break
        if "inputs_embeds" in model_kwargs:
            return np.ones((batch_size, 0), dtype=np.int32)
        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")
        return np.ones((batch_size, 1), dtype=np.int32) * bos_token_id

    def _prepare_attention_mask_for_generation(self,
        inputs: np.ndarray,
        pad_token_id: Optional[int],
        eos_token_id: Optional[int],
    ) -> np.ndarray:
        # No information for attention mask inference -> return default attention mask
        default_attention_mask = np.ones(inputs.shape[:2], dtype=np.int32)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs.shape) == 2 and np.issubdtype(inputs.dtype, np.integer)
        if not is_input_ids:
            return default_attention_mask

        is_pad_token_in_inputs = (pad_token_id is not None) and np.any(inputs == pad_token_id)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or not np.any(eos_token_id == pad_token_id)

        can_infer_attention_mask = is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = (inputs != pad_token_id).astype(np.int32)

        attention_mask = (
            attention_mask_from_padding * can_infer_attention_mask +
            default_attention_mask * (not can_infer_attention_mask)
        )
        return attention_mask

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, np.ndarray],
        decoder_start_token_id: int = None,
        bos_token_id: int = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""

        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        decoder_input_ids = model_kwargs.pop("decoder_input_ids", model_kwargs.pop("input_ids", None))

        # 2. Ensure that `decoder_start_token_id` equals `bos_token_id`
        assert decoder_start_token_id == bos_token_id, (
            "Make sure that `decoder_start_token_id` is correctly defined and that it is the same as `bos_token_id`."
            "Otherwise, the model will not behave as expected."
        )

        decoder_input_ids_start = np.ones((batch_size * self.decoder.num_codebooks, 1), dtype=np.int32) * decoder_start_token_id

        # 3. no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start

        # 4. user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id
        elif not np.all(decoder_input_ids[..., 0] == decoder_start_token_id):
            decoder_input_ids = np.concatenate([decoder_input_ids_start, decoder_input_ids], axis=-1)

            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = np.concatenate(
                    (np.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    axis=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs
    
    def _merge_criteria_processor_list(self, default_list: Union[StoppingCriteriaList], custom_list: Union[StoppingCriteriaList]) -> Union[StoppingCriteriaList]:
        if custom_list is None:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    raise ValueError(
                        f"A custom stopping criteria of type {type(custom)} has already been created. "
                        f"Default values of the stopping criteria should be passed as arguments instead of using a custom criteria."
                    )
        default_list.extend(custom_list)
        return default_list

    def get_stopping_criteria(self, generation_config, stopping_criteria: Optional[StoppingCriteriaList], tokenizer=None, **kwargs) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()

        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(MaxLengthCriteria(
                max_length=generation_config.max_length,
                max_position_embeddings=max_position_embeddings
            ))

        if generation_config.stop_strings is not None:
            if tokenizer is None:
                raise ValueError("A tokenizer is required when generating with stop strings.")
            criteria.append(StopStringCriteria(stop_strings=generation_config.stop_strings, tokenizer=tokenizer))

        if generation_config._eos_token_tensor is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config._eos_token_tensor))

        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria
    
    def get_logits_warper(self, generation_config) -> LogitsProcessorList:
        warpers = LogitsProcessorList()
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=1))
        return warpers

    def prepare_inputs_for_generation(
        self,
        input_ids,
        k_cache_list=None,
        v_cache_list=None,
        cache_lenth=None,
        attention_mask=None,
        head_mask=None,
        decoder_attention_mask=None,
        decoder_head_mask=None,
        prompt_hidden_states=None,
        prompt_attention_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        decoder_delay_pattern_mask=None,
        modality=None,
        **kwargs,
    ):
        if modality == "text":
            if k_cache_list is not None:
                input_ids = input_ids[:, -1:]
                prompt_hidden_states = None
            if input_ids.shape[-1] == 1:
                prompt_hidden_states = self.embed_prompts_cache([input_ids])[0]
            else:
                prompt_hidden_states = self.embed_prompts([input_ids])[0]
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "k_cache_list": k_cache_list,
                "v_cache_list": v_cache_list,
                "cache_lenth": cache_lenth,
                "decoder_input_ids": None,
                "attention_mask": attention_mask,
                "decoder_attention_mask": None,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "prompt_hidden_states": prompt_hidden_states,
                "prompt_attention_mask": prompt_attention_mask,
                "speaker_embedding": kwargs.get("speaker_embedding"),
                "modality": modality,
            }

        if modality == "audio":
            if decoder_delay_pattern_mask is None:
                input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
                    input_ids,
                    bos_token_id=self.generation_config.bos_token_id,
                    pad_token_id=self.generation_config.pad_token_id,
                    max_length=self.generation_config.max_length,
                )
            input_ids = self.decoder.apply_delay_pattern_mask(input_ids, decoder_delay_pattern_mask)
            input_ids = input_ids[:, -1:]
            prompt_hidden_states = None
            return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "encoder_outputs": encoder_outputs,
                "k_cache_list": k_cache_list,
                "v_cache_list": v_cache_list,
                "cache_lenth": cache_lenth,
                "decoder_input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_attention_mask": decoder_attention_mask,
                "head_mask": head_mask,
                "decoder_head_mask": decoder_head_mask,
                "cross_attn_head_mask": cross_attn_head_mask,
                "prompt_hidden_states": prompt_hidden_states,
                "prompt_attention_mask": prompt_attention_mask,
                "speaker_embedding": kwargs.get("speaker_embedding"),
                "modality": modality,
            }

    def generate(
        self,
        inputs: Optional[np.ndarray] = None,
        speaker_embedding: Optional[np.ndarray] = None,
        generation_config: Optional[Config] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs,
    ):
        # 1. Handle `generation_config` and kwargs that might update it, and validate the resulting objects
        if generation_config is None:
            generation_config = self.generation_config
        model_kwargs = kwargs

        # 2. Set generation parameters if not already defined
        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["speaker_embedding"] = speaker_embedding
        requires_attention_mask = True
        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
        )

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            print(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) "
                "to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation."
            )
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                print(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length

        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(
                f"Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than"
                f" the maximum length ({generation_config.max_length})"
            )
        if input_ids_seq_length >= generation_config.max_length:
            print(
                f"Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_new_tokens`."
            )

        input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config.bos_token_id,
            pad_token_id=generation_config.pad_token_id,
            max_length=generation_config.max_length,
        )
        # stash the delay mask so that we don't have to recompute in each forward pass
        model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask
        model_kwargs["decoder_input_ids"] = input_ids

        # set original modality to text
        model_kwargs["modality"] = "text"

        # 7. determine generation mode
        # 8. prepare batched CFG externally (to enable coexistance with the unbatched CFG)
        generation_config._eos_token_tensor = None

        # 9. prepare distribution pre_processing samplers
        logits_processor = LogitsProcessorList()

        # 10. prepare stopping criteria
        stopping_criteria = self.get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)

        # 11. prepare logits warper
        logits_warper = self.get_logits_warper(generation_config)
        model_kwargs["decoder_input_ids"] = input_ids

        # 12. run sample
        outputs = self._sample(
            model_kwargs["prompt_input_ids"],
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
            **model_kwargs,
        )
        output_ids = outputs[0]
        output_text_ids = outputs[1]

        # apply the pattern mask to the final ids
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs["decoder_delay_pattern_mask"])

        # revert the pattern delay mask by filtering the eos and bos token ids from the delay pattern mask
        _, mask = self.decoder.build_delay_pattern_mask(
            input_ids,
            bos_token_id=generation_config.bos_token_id,
            pad_token_id=generation_config.pad_token_id,
            max_length=output_ids.shape[1],
        )
        mask = (mask != generation_config.bos_token_id) & (mask != generation_config.pad_token_id)
        output_ids = output_ids[mask].reshape(batch_size, self.decoder.num_codebooks, -1)

        # append the frame dimension back to the audio codes
        output_ids = output_ids[None, ...]

        audio_scales = model_kwargs.get("audio_scales")
        if audio_scales is None:
            audio_scales = [None] * batch_size

        decode_sequentially = (
            generation_config.bos_token_id in output_ids
            or generation_config.pad_token_id in output_ids
            or generation_config.eos_token_id in output_ids
        )
        if not decode_sequentially:
            print("output_ids.shape:", output_ids.shape)
            raise ValueError("Please retry")
        else:
            output_values = []
            for sample_id in range(batch_size):
                sample = output_ids[:, sample_id]
                sample_mask = (sample >= 1024).sum(axis=(0, 1)) == 0
                if sample_mask.sum() > 0:
                    sample = sample[:, :, sample_mask]
                    sample = np.pad(sample, ((0, 0), (0, 0), (0, 512 - sample.shape[-1])), mode="edge")
                    sample = sample[None, ...]
                    print("sample.shape:", sample.shape)
                    sample = self.audio_encoder([sample])[0]
                    output_values.append(np.transpose(sample, (0, 2, 1)))
                else:
                    output_values.append(np.zeros((1, 1, 1)))
            # TODO: we should keep track of output length as well. Not really straightfoward tbh
            # output_values = pad_sequence(output_values, padding_value=0)[..., 0]
            output_values = output_values[0][..., 0]

        return output_values, output_text_ids

    def _sample(
        self,
        input_ids: np.array,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id

        eos_token_id = self.generation_config.eos_token_id
        stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        scores = None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = np.ones(batch_size, dtype=np.int32)

        decoder_input_ids = model_kwargs.get("decoder_input_ids", None)

        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            lm_logits, k_cache_list, v_cache_list, cache_lenth = self(**model_inputs)

            next_token_logits = lm_logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(next_token_logits)
            next_token_scores = logits_warper(next_token_scores)

            # sample
            next_tokens = self.sample_output.run(None, {"next_token_scores": next_token_scores})[0]

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                # next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                        
            # update generated ids, model inputs, and length for next step
            input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=-1)
            if input_ids[0][-1] == 151654:
                model_kwargs["prompt_input_ids"] = input_ids
                input_ids = decoder_input_ids
                model_kwargs["modality"] = "audio"
   
            model_kwargs = self.update_model_kwargs_for_generation(
                k_cache_list, v_cache_list, cache_lenth,
                model_kwargs
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        return input_ids, model_kwargs["prompt_input_ids"]

    def update_model_kwargs_for_generation(self,
            k_cache_list,
            v_cache_list,
            cache_lenth: int,
            model_kwargs: Dict[str, Any],
        ) -> Dict[str, Any]:
            # update past_key_values keeping its naming used in model code
            model_kwargs["k_cache_list"] = k_cache_list
            model_kwargs["v_cache_list"] = v_cache_list
            model_kwargs["cache_lenth"] = cache_lenth
            return model_kwargs