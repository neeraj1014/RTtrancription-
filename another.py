from pyannote.audio import Pipeline
key = "huggingface_auth_key"
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=key)


# apply the pipeline to an audio file
diarization = pipeline(r"C:\Users\Neeraj\Desktop\model\audio\2830-3980-0043.wav")

# # dump the diarization output to disk using RTTM format
# with open("audio.rttm", "w") as rttm:
#     diarization.write_rttm(rttm)
print(diarization)

