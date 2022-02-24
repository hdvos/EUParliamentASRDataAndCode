import torch
import time
import os
import shutil
import sox
from pprint import pprint, pformat
from dataclasses import dataclass, asdict
from collections import namedtuple
import logging
from pydub import AudioSegment
import json

#TODO figure out what to do with logging.
logging.basicConfig(filename='diarization.log', level=logging.INFO)

@dataclass      #TODO: myturncontainer in utils script
class MyTurnContainer:
    start: float
    end : float
    speaker: str
    turn_i: int

@dataclass
class DiarizationResults:
    original_wavfile:str
    original_turns:list
    my_turns:list


# Dataclasses for bookkeeping when splitting wavfiles
@dataclass
class BookKeep:
    original_file:str
    segments:list

DiarizationBookkeep = BookKeep # Renaming this to avoid ambiguity in the pipeline but not renaming every occerence in this script.

@dataclass
class BookKeepSegment(MyTurnContainer):
    filename:str = ''

DiarizationBookkeepSegment = BookKeepSegment # Renaming this to avoid ambiguity in the pipeline but not renaming every occerence in this script.

# adapted from https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/pipeline 
class MyDiarizer():
    """Diarize a wavfile.
    """
    def __init__(self, wavfile:str,  save_rttm:bool = True, filebase = ''): #TODO: parametrize model choice
        """Initialize MyDiarizer.

        :param wavfile: The name of the wavfile that needs to be diarized
        :type wavfile: str
        :param save_rttm: Save the rttm file of the diarization to disk, defaults to True
        :type save_rttm: bool, optional
        """
        assert os.path.exists(wavfile), f'{wavfile} does not exist.'

        start_load_model = time.time()
        self.pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
        end_load_model = time.time()
        logging.info(f"Loading model took {end_load_model - start_load_model:.2f} seconds")

        self.wavfile = wavfile
        if not filebase:
            self.filebase = self._get_filebase()
        else:
            self.filebase = filebase

        self.save_rttm = save_rttm

    def _get_filebase(self) -> str:
        """get the base of the wavfile (basename minus extension). This will be used to generate the names of the output files.

        :return: wavfile base
        :rtype: str
        """
        basename = os.path.basename(self.wavfile)
        filebase = basename.split('.')[0]
        # self.filebase = filebase

        return filebase

    def _post_process(self, minturnlength:float = 2.0, max_silence_length:float = 3.0) -> list:
        """Post process the output of pyannote. Mainly apply some smoothing.

        :param minturnlength: The minimal length a turn should have in seconds, defaults to 2.0
        :type minturnlength: float, optional
        :param max_silence_length: The maximum length of silence within a turn, defaults to 3.0
        :type max_silence_length: float, optional
        :return: a list of turns.
        :rtype: list
        """

        # Define some initial versions to before the loop.
        prevspeaker = 'A'
        prev_end = 0
        turn_nr = 1
        turns = []
        current_turn = MyTurnContainer(0,0,'A', turn_nr)

        for turn, _, speaker in self.diarization.itertracks(yield_label=True):
            turnlength = turn.end - turn.start

            # If the duration between current and previous turn is smaller than the max silence length, AND the speaker in both turns is the same, 
            # then add turn to previous turn
            if speaker == prevspeaker and (turn.start - prev_end) < max_silence_length:
                current_turn.end = turn.end
            # If the current turn is too short to be a turn, it is added to the previous turn.
            elif turn.end-turn.start  < minturnlength:
                current_turn.end = turn.end
            # Make a cut between current turn and previous turn.
            else:
                turns.append(current_turn)
                turn_nr += 1
                current_turn = MyTurnContainer(turn.start,turn.end,speaker, turn_nr)

            prevspeaker = speaker
            prev_end = turn.end     #TODO: <-- redundant, can be replaced with current_turn.end

        turns.append(current_turn)
        return turns

    def split_wavfile(self, output_folder: str, bookkeepfile:str, bookkeepformat:str='json', use_myturn:bool=True, include_silences:bool = False):
        """Split a wavfile into turns determined during diarization. Note: function assumes that self.diarize has already run

        :param output_folder: folder in which the resulst of the diarization will be stored
        :type output_folder: str
        :param bookkeepfile: the name of the file where the bookkeeping will be stored.
        :type bookkeepfile: str
        :param bookkeepformat: Currently not implemented. In future add csv., defaults to 'json'
        :type bookkeepformat: str, optional
        :param use_myturn: use smoothed turns, else the original turns as provided by pyannote will be given, defaults to True
        :type use_myturn: bool, optional
        :param include_silences: Not yet implemented. Also include the silences between two turns into the new wavfiles, defaults to False
        :type include_silences: bool, optional
        :raises NotImplementedError: Some functionalities have no priority, but might be interesting in the future.
        :return: BookKeep
        :rtype: a dataclass containing metadata on the created files.
        """
        if not bookkeepformat == 'json':
            raise NotImplementedError("No other types than json are supported atm")
        if include_silences:
            raise NotImplementedError("Feature \"include_silences\" has not yet been implemented.")

        logging.info(f"Split {self.wavfile}")
        if include_silences:
            raise NotImplementedError()

        if use_myturn:
            turns = self.myturns
        else:
            turns = [_ for _ in self.diarization.itertracks(yield_label=True)]


        original_wav = AudioSegment.from_wav(self.wavfile)
        
        bookkeep = BookKeep(original_file=self.wavfile, segments=[])
        
        for i, turn in enumerate(turns, start=1):
            output_folder2 = os.path.join(output_folder, self.filebase)

            if not os.path.exists(output_folder2):
                os.makedirs(output_folder2)

            outfilename = os.path.join(output_folder, f'{i:03d}_{turn.speaker}_{self.filebase}.wav')
            newAudio = original_wav[turn.start*1000:turn.end*1000]
            
            newAudio.export(outfilename, format="wav") #Exports to a wav file in the current path.

            bks = BookKeepSegment(start=turn.start, end=turn.end, speaker=turn.speaker, filename=outfilename, turn_i=i)
            bookkeep.segments.append(bks)

        with open(bookkeepfile, 'wt') as fo:
            # print(asdict(bookkeep))
            json.dump(asdict(bookkeep), fo)
            logging.info(f"bookkeepfile saved at: {bookkeepfile}")
    
        return bookkeep

    def diarize(self, save_original_turns = False, save_myturns = True) -> DiarizationResults:
        """Apply Pyannote diarization.

        :param save_original_turns: whether to save the original turns as wavfiles (Notimplemented), defaults to False #TODO: remove option
        :type save_original_turns: bool, optional 
        :param save_myturns: save the smoothed turns as wavfiles, defaults to False #TODO: remove option
        :type save_myturns: bool, optional
        :return: a dataclass containing information about the diarization process.
        :rtype: DiarizationResults
        """
        logging.info(f"Diarizing {self.wavfile}. Duration: {sox.file_info.duration(self.wavfile):.2f}")
        
        start_time = time.time()
        self.diarization = self.pipeline({'audio': self.wavfile})
        end_time = time.time()
        logging.info(f"Diarizing took {end_time-start_time:.2f} seconds")
        
        self.original_turns = []
    
        for i, row in enumerate(self.diarization.itertracks(yield_label=True), 1):
            self.original_turns.append(MyTurnContainer(row[0].start, row[0].end, row[2], i))
        
        # pprint(self.original_turns)
        if self.save_rttm:
            rttm_name = f'{self.filebase}.rttm'  #TODO: include rttm filename as paramtere somewhere.
            with open('test2.rttm', 'wt') as f:
                self.diarization.write_rttm(f)
        
    
        self.myturns = self._post_process()
        # for newturn in self.myturns:
        #     print(newturn.__dict__)

        diarization_results = DiarizationResults(self.wavfile, self.original_turns, self.myturns)

        return diarization_results
        

if __name__ == "__main__":
    # print("Hello world")
    # print(os.listdir("/home/hugo/MEGA/work/ASR/build_pipeline"))
    testwav = '/home/hugo/MEGA/work/ASR/build_pipeline/testfile2.wav'
    assert os.path.exists(testwav)
    diarizer = MyDiarizer(testwav)
    diarizer.diarize()
    diarizer.split_wavfile('/home/hugo/MEGA/work/ASR/build_pipeline/diarization/test_wav_out', 'test_bookkeep.json')
    with open('diarizerobject.txt', 'wt') as out:
        diarizer_string = pformat(dir(diarizer.diarization))
        out.write(diarizer_string)