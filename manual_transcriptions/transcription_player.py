import pandas as pd
import sys
from pydub import AudioSegment
from pydub.playback import play
import os

def get_audiofiles(csv='semi_manual_transcripts.csv'):
    data_df = pd.read_csv(csv, sep='\t')
    return data_df.wavfile_location.values
    

if __name__ == "__main__":
    try:
        startrow = int(sys.argv[1]) - 1
    except IndexError:
        startrow = 0

    wavfiles_list = get_audiofiles()
    prompt = 'Type n for next soundfile, press enter to play (again)'
    for wavfile in wavfiles_list[startrow:]:
        assert os.path.exists(wavfile), f"{wavfile} does not exist"
        print("*"*50)
        print(f"Playing {os.path.basename(wavfile)}")
        while True:
            fragment = AudioSegment.from_wav(wavfile)
            play(fragment)
            
            if input(prompt).lower() == 'n':
                break
            else:
                continue
