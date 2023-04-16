from csv import DictReader
# from logging import DEBUG
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
import librosa
import torch
import jiwer
import pywer
import json
import pandas as pd
import torchaudio
import re
from pyctcdecode import Alphabet, BeamSearchDecoderCTC
from pyctcdecode import build_ctcdecoder
import kenlm
from dataclasses import dataclass
import numpy as np
import pandas as pd
from functools import partial

DEBUG = False

if DEBUG:
    TRANSCRIPTIONS_FILE = "/home/hugo/MEGA/work/ASR/EUParliamentASR/evaluation/transcriptions/manual_transcriptions.csv"
    SOUNDFILES_FOLDER = "/home/hugo/MEGA/work/ASR/EUParliamentASR/evaluation/transcriptions/manual_transcriptions_soundfiles"
    HOTWORDS_COUPLE_FILE = "/home/hugo/MEGA/work/ASR/EUParliamentASR/wavfiles_with_entities.csv"
    KENLM_MODEL = '/home/hugo/MEGA/work/ASR/EUParliamentASR/kenlm_models/europarl-5-gram.bin'
    WAV2VEC_MODEL = "facebook/wav2vec2-base-10k-voxpopuli-ft-en"
else:
    TRANSCRIPTIONS_FILE = "/data/voshpde/pipeLine/evaluation/transcriptions/manual_transcriptions.csv"
    SOUNDFILES_FOLDER = "/data/voshpde/pipeLine/evaluation/transcriptions/manual_transcriptions_soundfiles"
    HOTWORDS_COUPLE_FILE = "../wavfiles_with_entities.csv"
    KENLM_MODEL = '../kenlm_models/europarl-5-gram.bin'
    WAV2VEC_MODEL = "facebook/wav2vec2-base-10k-voxpopuli-ft-en"


# KENLM_MODEL = '/home/hugo/MEGA/work/ASR/EUParliamentASR/kenlm_models'

EXPERIMENT_WAVS = ''



# @dataclass
# class Models():
#     wav2vec_model:object
#     kenlm_model:object

def validate_soundfiles(transcriptions_df:pd.DataFrame):
    for soundfile in transcriptions_df.Sound_file:
        if not os.path.exists(soundfile):
            raise FileNotFoundError(f"Could not locate {soundfile}")

def get_id_from_wavfile_hotwords(wavfile:str):
    basename = os.path.basename(wavfile)
    wavfile_id = os.path.splitext(basename)[0]
    assert len(wavfile_id) == 8 or len(wavfile_id) == 10
    return wavfile_id



def load_hotwords(df:pd.DataFrame, hotwords_file:str):
    assert os.path.exists(hotwords_file)
    hotwords_df = pd.read_csv(hotwords_file, delimiter='\t', index_col=0)

    hotwords_df['identifier'] = hotwords_df.wavfile.apply(get_id_from_wavfile_hotwords)

    return hotwords_df

def validate_identifier(identifier:str):
    if not (len(identifier) == 8 or len(identifier) == 8):
        raise ValueError(f"{identifier} is not a valid identifier")


def retrieve_hotwords(identifier, hotwords):
    row = hotwords[hotwords.identifier == identifier]
    entities_string = row.entities.values[0]

    if not entities_string:
        return entities_string
    else:
        return str(entities_string)
    # print(entities_string)
    


def read_data(transcriptions_file:str, soundfiles_folder:str, hotwords_file:str):
    assert os.path.exists(transcriptions_file), f"{transcriptions_file} does not exist"

    transcriptions_df = pd.read_csv(transcriptions_file, delimiter='\t')
    transcriptions_df.Sound_file = transcriptions_df.Sound_file.apply(lambda x:x.strip('.pcm'))
    transcriptions_df.Sound_file = transcriptions_df.Sound_file.apply(lambda x:os.path.join(soundfiles_folder, x))
    validate_soundfiles(transcriptions_df)

    transcriptions_df['identifier'] = transcriptions_df.Sound_file.apply(lambda x:os.path.basename(x))
    transcriptions_df['identifier'] = transcriptions_df.identifier.apply(lambda x:os.path.splitext(x)[0])
    transcriptions_df['identifier'] = transcriptions_df.identifier.apply(lambda x:x.split('_')[-1])
    transcriptions_df.identifier.apply(validate_identifier)

    hotwords_df = load_hotwords(transcriptions_df, hotwords_file)
    # hotwords_retriever = partial(retrieve_hotwords, )
    
    transcriptions_df['hotwords'] = transcriptions_df.identifier.apply(retrieve_hotwords, hotwords=hotwords_df)
    return transcriptions_df

def transcribe(row, processor, w2vmodel, decoder, hotwords_weight):
    filename = row.Sound_file
    hotwords = row.hotwords
    if hotwords == 'nan':
        hotwords = ''
    else:
        hotwords = hotwords.split('|')

    speech, rate = librosa.load(filename, sr=16000)
    input_values = processor(speech, return_tensors = 'pt').input_values
    #Store logits (non-normalized predictions)
    logits = w2vmodel(input_values).logits.cpu().detach().numpy()[0]

    if hotwords:
        transcription = decoder.decode(logits, 
            hotwords=hotwords,
            hotword_weight=hotwords_weight,)
    else:
        transcription = decoder.decode(logits)

    return transcription

def inference(transcriptions_df, processor, model, kenlm_model, alpha, beta, hotwords_weight):
    
    vocab_list = list(processor.tokenizer.get_vocab().keys())
    with open("vocab_list_TEST.txt", 'wt') as outtt:
        outtt.write(str(vocab_list))
    # vocab_list[0] = ""
    vocab_list[0] = " "
    # # replace special characters
    vocab_list[1] = "⁇"
    vocab_list[2] = "⁇"
    vocab_list[3] = "⁇"
    # # convert space character representation
    vocab_list[4] = " "

    labels = vocab_list
    with open("vocab_list_after.txt", 'wt') as outtt:
        outtt.write(str(vocab_list))
    # build the decoder and decode the logits
    # decoder = BeamSearchDecoderCTC(alphabet)
    decoder = build_ctcdecoder(
        labels,
        kenlm_model,
        ctc_token_idx = 0,
        alpha=alpha,  # tuned on a val set
        beta=beta,  # tuned on a val set
    )

    experiment_transcriptions = transcriptions_df.copy()
    experiment_transcriptions['inference'] = experiment_transcriptions.apply(transcribe, axis=1, processor=processor, w2vmodel=model, decoder=decoder, hotwords_weight=hotwords_weight)
    # inference = []
    
    if not os.path.exists('results'):
        os.makedirs('results')

    outfilename = f"results/experiment_{str(hotwords_weight).replace('.', '_')}.csv"
    experiment_transcriptions.to_csv(outfilename, sep='\t')




def main():
    weights = np.arange(3,6,1)
    print(f"Load data ... ", end='')
    data_df = read_data(TRANSCRIPTIONS_FILE, SOUNDFILES_FOLDER, HOTWORDS_COUPLE_FILE)
    print(u'\u2713')
            
    print(f"Load kenlm model ... ", end='')
    kenlm_model = kenlm.Model(KENLM_MODEL)
    print(u'\u2713')

    print(f"Load transformer models ... ", end='')
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL, cache_dir="/data/voshpde/transformers_models")
    model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL, cache_dir="/data/voshpde/transformers_models")
    print(u'\u2713')

    for weight in weights:
        with open("started.txt", 'a') as started:
            started.write(f"{weight}\n")
        inference(data_df, processor, model, kenlm_model, alpha=0.1, beta=0.2, hotwords_weight=weight)
        with open("finished.txt", 'a') as finished:
            finished.write(f"{weight}\n")

if __name__ == "__main__":
    main()