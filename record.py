import pyaudio
import sys
import time 
import numpy as np 
from real_time_app.load_models import base_model, stt_model, osd_model,vad_model,diarization_model,speech_quality
from real_time_app.helper import get_np_n_torch_array
from real_time_app.dummy import check

KEY = "huggingface_auth_key"
INT16_MAX_ABS_VALUE = 32768.0

# Parameters for audio capture
CHUNK = int(3*16000)  # Number of audio frames per buffer
SAMPLE_FORMAT = pyaudio.paInt16  # 16 bits per sample
CHANNELS = 1  # Mono channel (set to 2 for stereo if needed)
FS = 16000  # Sampling frequency (samples per second)

BASE_MODEL = base_model(KEY)
SPEECH_DETECTING_MODEL = vad_model(BASE_MODEL)
OVERLAP_DETECTING_MODEL = osd_model(BASE_MODEL)
TRANSCRIBING_MODEL = stt_model()
# DIARIZATION_MODEL = diarization_model(KEY)
SPEECH_Q = speech_quality(KEY)

print("All models loaded!!!")

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open a new stream for audio recording
stream = p.open(format=SAMPLE_FORMAT,
                channels=CHANNELS,
                rate=FS,
                frames_per_buffer=CHUNK,
                input=True)

print('Mic started working!!!')

try:
    while True:
        st = time.time()
        data = stream.read(CHUNK)
        torch_arr_dict,np_arr = get_np_n_torch_array(data)
        vad = SPEECH_DETECTING_MODEL(torch_arr_dict)
        if list(vad.itertracks(yield_label=True)):
            osd = OVERLAP_DETECTING_MODEL(torch_arr_dict)
            if not list(osd.itertracks(yield_label=True)):
                val = np.mean([turn.end-turn.start for turn, _ , _ in vad.itertracks(yield_label=True)])
                if val < 1:
                    print("!!!It's a short speech!!!")
                else:
                    Segment, _ = TRANSCRIBING_MODEL.transcribe(np_arr, language='en')
                    text = " ".join(seg.text for seg in Segment)
                    if text=="":
                        print(list(vad.itertracks(yield_label=True)))
                    else:
                        print(text)

                    # output = SPEECH_Q(torch_arr_dict)
                    # check(output)
            else:
                # diarization, source = DIARIZATION_MODEL(torch_arr_dict)
                output = SPEECH_Q(torch_arr_dict)
                C50,SNR = check(output)
                if C50 > 10 and SNR > 0 :
                    print("#####Resloving overlapping speech#####",C50,SNR)
                    Segment, _ = TRANSCRIBING_MODEL.transcribe(np_arr, language='en')
                    text = " ".join(seg.text for seg in Segment)
                    if text=="":
                        print(list(vad.itertracks(yield_label=True)))
                    else:
                        print(text)
                else:
                    print("!!!Many of you are speaking at same time, speak one at a time!!!")
        else:
            print("!!No one is speaking!!!")
except KeyboardInterrupt:
    print("Stream interrupted by user. Exiting...")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Ensure the stream is properly closed
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Stream closed and PyAudio terminated.")


sys.exit(0)

