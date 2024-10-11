import numpy as np 
import torch 
INT16_MAX_ABS_VALUE = 32768.0 


def get_np_n_torch_array(data, fs=16000, get=True):
    np_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    waveform = torch.from_numpy(np_data).float()/INT16_MAX_ABS_VALUE
    # Create the "waveform" and "sample_rate" mapping
    audio_mapping = {
                "waveform": torch.unsqueeze(waveform, 0),  # Tensor of shape (channel, time)
                "sample_rate": fs
            }
    if get == True:
        return audio_mapping, np_data/INT16_MAX_ABS_VALUE
    else:
        return audio_mapping