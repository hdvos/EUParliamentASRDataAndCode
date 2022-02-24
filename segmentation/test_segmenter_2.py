from audio_segmenter_3 import segment_wavfile
import os,sys

TESTFILE  = '../testfile2.wav'
OUTFOLDER = 'testout'

if __name__ == "__main__":
    os.chdir(os.path.split(__file__)[0])
    
    assert os.path.exists(TESTFILE), 'testfile should exist'

    if not os.path.exists(OUTFOLDER):
        os.makedirs(OUTFOLDER)

    segment_wavfile(TESTFILE, OUTFOLDER)