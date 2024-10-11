from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
from faster_whisper import WhisperModel

def stt_model(model_size = "base", device ="cpu"):
    if device == 'cpu':
        # Run on CPU with INT8
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    elif device == 'cuda'or device == 'gpu':
        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    else:
        print("Error!!!!")
        
    return model



def base_model(key):
    model = Model.from_pretrained(
    "pyannote/segmentation-3.0", 
    use_auth_token=key)
    return model


def vad_model(base_model):
    pipeline = VoiceActivityDetection(segmentation=base_model)
    HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
    "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    return pipeline

def osd_model(base_model):
    pipeline = OverlappedSpeechDetection(segmentation=base_model)
    HYPER_PARAMETERS = {
    # remove overlapped speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-overlapped speech regions shorter than that many seconds.
    "min_duration_off": 0.0
    }
    pipeline.instantiate(HYPER_PARAMETERS)
    return pipeline

def diarization_model(key):
    model = Model.from_pretrained(
    "pyannote/separation-ami-1.0", 
    use_auth_token=key)
    return model


def speech_quality(key):
    model = Model.from_pretrained("pyannote/brouhaha",
                                use_auth_token=key)
    inference = Inference(model)
    return inference