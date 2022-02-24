# load pipeline
import torch

WAV_PATH = '/data/voshpde/pyannote/20140701.wav'
RTTM_OUT = 'test_rttm.rttm'





if __name__ == "__main__"
    pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')

    # apply diarization pipeline on your audio file
    diarization = pipeline({'audio': WAV_PATH})

    # dump result to disk using RTTM format
    with open(RTTM_OUT, 'w') as f:
        diarization.write_rttm(f)
  
    # iterate over speech turns
    with open('readable_output.txt', 'wt') as out:
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            out.write(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.\n')
# Speaker "A" speaks between t=0.2s and t=1.4s.
# Speaker "B" speaks between t=2.3s and t=4.8s.
# Speaker "A" speaks between t=5.2s and t=8.3s.
# Speaker "C" speaks between t=8.3s and t=9.4s.
# ...
