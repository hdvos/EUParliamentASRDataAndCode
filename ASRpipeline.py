# from diarization.Diarization2 import MyDiarizer #, DiarizationBookkeep, DiarizationBookkeepSegment
from segmentation.audio_segmenter_3 import segment_wavfile
from transcription.Transcribe3 import Transcriber
from librosa import get_duration
from pprint import pprint
from dataclasses import dataclass, asdict
import os
from shutil import rmtree
import time
import json
import logging

def prepare_output_folder(foldername:str):
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    else:
        rmtree(foldername)


# def my_get_duration(wavname):
#     y, sr = librosa.load(librosa.ex('trumpet'))

import argparse

parser = argparse.ArgumentParser(description='Pre-process and transcribe audio files.')
parser.add_argument("wavfile", help="the wavfile that needs to be processed.")
parser.add_argument('-rttm_output', type=str, default=None, 
                    help="Where to store the diarization bookkeep file when it is created, or where to read it from if stage >=2")                                        
parser.add_argument("-segmentation_output_folder", type=str, default="segmentation_wavfiles",
                    help="the folder where the results of segmentation will be stored.")
parser.add_argument("-segmentation_bookkeep_file", type=str, default=None,
                    help="The json file with the segmentation bookkeep.")
parser.add_argument("-asr_model", type=str, default="facebook/wav2vec2-base-10k-voxpopuli-ft-en",
                    help="The asr model.")
parser.add_argument("-kenlm_model", type=str, default="/data/voshpde/pipeLine/kenlm_models/europarl-5-gram.bin",
                    help="The asr model.")
parser.add_argument("-transcriptions_output_folder", type=str, default="/data/voshpde/pipeLine/ASR_transcriptions/",
                    help="Folder where the transcriptions will be placed.")
parser.add_argument("-logfolder", type=str, default='/data/voshpde/pipeLine/logs',
                    help="The folder where the logfiles will go.")


#facebook/wav2vec2-base-100k-voxpopuli

args = parser.parse_args()



'%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s'


# print(args.accumulate(args.integers))


process_id = os.path.splitext(os.path.basename(args.wavfile))[0]

logfolder = args.logfolder
print(f"LOGFOLDER: {logfolder}")
if not os.path.exists:
    os.makedirs(logfolder)
logfile = os.path.join(logfolder, f'{process_id}.log')
print(f"LOGFILE: {logfile}")
logging.basicConfig(filename=logfile, level=logging.DEBUG, format='%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s', force=True)

logging.info(f"process ID: {process_id}")

WAVFILE = args.wavfile
logging.info(f'Processing: {WAVFILE}')


SEGMENTATION_OUTPUT_FOLDER = args.segmentation_output_folder
logging.info(f'segmentation output folder: {SEGMENTATION_OUTPUT_FOLDER}')

TRANSCRIPTIONS_OUTPUT_FOLDER =args.transcriptions_output_folder
logging.info(f'transcriptions output folder: {TRANSCRIPTIONS_OUTPUT_FOLDER}')
if not os.path.exists(TRANSCRIPTIONS_OUTPUT_FOLDER):
    os.makedirs(TRANSCRIPTIONS_OUTPUT_FOLDER)


def do_diarization(wavfile_path:str):
    assert os.path.exists(wavfile_path)

    logging.debug(f"do diarization of {wavfile_path}")

    diarizer = MyDiarizer(wavfile_path)
    diarization_results = diarizer.diarize()
  
    return diarization_results


# TODO: timeregistration dataclass
@dataclass
class time_registration:
    audio_duration:float
    diarization_duration:float
    segmentation_duration:float
    recognition_duration:float

if __name__ == "__main__":

    # timereg.write(f"Audiofile is {get_duration(filename = WAVFILE)} seconds.\n")
    diarization_start=time.time()
    logging.warning("NO DIARIZATION!")
    testwav = WAVFILE
    # diarization_results = do_diarization(testwav)
    # diarization_duration = time.time() - diarization_start
    
    # logging.info(f"Diarzation took {diarization_duration:.2f} seconds")

    segmentation_start = time.time()
    segments_metadata, segmentationbookkeep = segment_wavfile(testwav, SEGMENTATION_OUTPUT_FOLDER)
    segmentation_end = time.time()
    segmentation_duration = segmentation_end - segmentation_start

    logging.info(f"Segmentation took {segmentation_duration:.2f} seconds")

    transcribing_start = time.time()
    my_transcriber=Transcriber(args.asr_model, args.kenlm_model, kenlm_alpha=0.1, kenlm_beta=0.3)

    json_output = os.path.join(TRANSCRIPTIONS_OUTPUT_FOLDER, 'json')
    csv_output = os.path.join(TRANSCRIPTIONS_OUTPUT_FOLDER, 'csv')

    my_transcriber.TranscribeFromBookkeep(segmentationbookkeep, json_folder=json_output, csv_folder=csv_output)
    transcribing_end = time.time()
    transcribing_duration = transcribing_end - transcribing_start
    logging.info(f"transcribing took {transcribing_duration:.2f} seconds")

    total_runtime = time.time() - diarization_start
    logging.info(f"total runtime: {total_runtime:.2f}")
    
    with open("/data/voshpde/pipeLine/all_slurm4/processed_wavfiles.txt", 'a') as pw:
        pw.write(WAVFILE)
    # Transcribe.TranscribeFromBookkeep(segmentationbookkeep, 'voxpopuli_transcriptions_json', 'transcriptions_csv', "s")
    # # TODO Also implement reader for segmentation
    # # timereg.write(f"Stage 3 took {time.time() - stage_3_start:.2f} seconds.\n")

    # # TODO: recombine