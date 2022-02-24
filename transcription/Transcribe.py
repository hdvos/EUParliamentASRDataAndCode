import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import os,sys
from segmentation.Segmentation import LengthSegmentationBookkeep, LengthSegmentationBookkeepSegment
from dataclasses import dataclass, asdict
import json
from csv import writer

@dataclass
class InferenceResultSegment():
    wavfile:str

    absolute_start_seconds:float
    absolute_start_frames:int
    
    absolute_end_seconds:float
    absolute_end_frames:int

    diarization_turn_i:int
    time_segment_i:int

    speaker:str

    inferred_text:str

@dataclass
class InferenceResults():
    original_wavfile:str
    segments:list

# https://www.analyticsvidhya.com/blog/2021/02/hugging-face-introduces-the-first-automatic-speech-recognition-model-wav2vec2/

# testfile = "/data/voshpde/audiofiles/20140701.wav"
# testfile_resampled = "/data/voshpde/audiofiles/20140701_resampled.wav"


# # https://unix.stackexchange.com/questions/274144/sox-convert-the-wav-file-with-required-properties-in-single-command 
# os.system(f"sox {testfile} -r 16000 {testfile_resampled}")

# #+++++++++
# print('+++++++++load tokenizer+++++++++')
# tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
# print('+++++++++load model+++++++++')
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# print('+++++++++save tokenizer+++++++++')
# tokenizer.save_pretrained('/tokenizer/')
# print('+++++++++save model+++++++++')
# model.save_pretrained('/model/')



#load any audio file of your choice

MODEL_NAME = "facebook/wav2vec2-base-960h"
print("THE MODELNAME", MODEL_NAME)

#load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)


# with open("test_transcripts.txt", 'wt') as out:
#     out.write("/data/voshpde/asr_huggingface/transcriptions.txt")


def get_text(filename:str):
    try:
        speech, rate = librosa.load(filename, sr=16000)

        input_values = tokenizer(speech, return_tensors = 'pt').input_values
        #Store logits (non-normalized predictions)
        logits = model(input_values).logits

        #Store predicted id's
        predicted_ids = torch.argmax(logits, dim =-1)
        #decode the audio to generate text
        transcription = tokenizer.decode(predicted_ids[0])
        return transcription
    except ValueError:
        return ''

def transcribe_wavfile(filename:str):
    if not os.path.exists(filename):
        raise RuntimeError(f"Cannot transcribe {filename}. File does not exist.")

    text = get_text(filename)
    return text


    # print(transcriptions)

def results_to_json(results:InferenceResults, filename:str):
    if not filename.endswith(".json"):
        raise ValueError(f"Filename must have the .json extension")

    with open(filename, 'wt') as out:
        json.dump(asdict(results), out)

    print(f"Wrote json to {filename}")


def results_to_csv(results:list, filename:str):
    if not filename.endswith(".csv"):
        raise ValueError(f"Filename must have the .csv extension")

    with open(filename, 'wt') as out:
        mywriter = writer(out, delimiter = '\t')
        mywriter.writerow(["speaker", 'inferred_text', 'duration' , 'absolute_start_seconds', 'absolute_end_seconds', 'segment_wavfile', 'original_wavfile'])
        for result in results.segments:
            print([result.speaker, result.inferred_text, result.absolute_end_seconds-result.absolute_start_seconds , result.absolute_start_seconds, result.absolute_end_seconds, result.wavfile, results.original_wavfile])
            mywriter.writerow([result.speaker, result.inferred_text, result.absolute_end_seconds-result.absolute_start_seconds , result.absolute_start_seconds, result.absolute_end_seconds, result.wavfile, results.original_wavfile])

    print(f"Wrote csv to {filename}")

def make_json_filename(results:InferenceResults, folder = './') -> str:
    basename = os.path.basename(results.original_wavfile)
    naked_basename = os.path.splitext(basename)[0]
    json_filename = f"{naked_basename}.json"
    json_filename = os.path.join(folder, json_filename)
    return json_filename

def make_csv_filename(results:InferenceResults, folder = './') -> str:
    basename = os.path.basename(results.original_wavfile)
    naked_basename = os.path.splitext(basename)[0]
    csv_filename = f"{naked_basename}.csv"
    csv_filename = os.path.join(folder, csv_filename)
    return csv_filename


def TranscribeFromBookkeep(segmentationbookkeep:LengthSegmentationBookkeep, json_folder='./json_output', csv_folder='./csv_output'):
    print("THE MODELNAME TRANSCRIBE FROM BOOKKEEP", MODEL_NAME)
    if not os.path.exists(json_folder):
        os.makedirs(json_folder)
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    
    results = InferenceResults(
        original_wavfile = segmentationbookkeep.original_filename,
        segments = []
    )

    for segment in segmentationbookkeep.segments:
        text = transcribe_wavfile(segment.time_segmented_filename)
        inference_result = InferenceResultSegment(
            wavfile = segment.time_segmented_filename,
            
            absolute_start_seconds = segment.absolute_start_seconds,
            absolute_start_frames = segment.absolute_start_frames,
            
            absolute_end_seconds = segment.absolute_end_seconds,
            absolute_end_frames = segment.absolute_end_frames,

            diarization_turn_i = segment.diarization_turn_i,
            time_segment_i = segment.time_segment_i,

            speaker = segment.speaker,
            inferred_text = text

        )
        results.segments.append(inference_result)

    json_filename = make_json_filename(results, json_folder)
    csv_filename = make_csv_filename(results, csv_folder)

    results_to_json(results, json_filename)
    results_to_csv(results, csv_filename)
