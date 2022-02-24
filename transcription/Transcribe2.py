import csv
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, Wav2Vec2Processor
import os,sys
from segmentation.Segmentation import LengthSegmentationBookkeep, LengthSegmentationBookkeepSegment
from dataclasses import dataclass, asdict
import json
from csv import writer
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, build_ctcdecoder
import kenlm
import logging 
from pprint import pformat


@dataclass
class InferenceResultSegment():
    wavfile:str

    start_seconds:float
    start_frames:int
    
    end_seconds:float
    end_frames:int

    # diarization_turn_i:int
    identifier:int

    # speaker:str

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

# MODEL_NAME = "facebook/wav2vec2-base-960h"
# print("THE MODELNAME", MODEL_NAME)




#load model and tokenizer
# asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained(MODEL_NAME)
# asr_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
# asr_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

class Transcriber(object):

    def __init__(self, model_name:str, kenlm_model_filename:str = '', kenlm_alpha:float=0.1, kenlm_beta:float=0.3) -> None:
        super().__init__()
        self.model_name = model_name
        self.asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
        logging.info(f"Tokenizer loaded: {model_name}")
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        logging.info(f"asr model loaded: {model_name}")
        self.asr_processor = Wav2Vec2Processor.from_pretrained(model_name)
        logging.info(f"asr processor loaded: {model_name}")

        self.kenlm_model_filename = kenlm_model_filename
        if kenlm_model_filename:
            self._prepare_lm(kenlm_model_filename, alpha=kenlm_alpha, beta=kenlm_beta)
            logging.info(f"kenlm model prepared: {kenlm_model_filename}")
        
        logging.info("Transcriber loaded")
        logging.debug(f"Model name: {model_name}")
        logging.debug(f"KenLM model: {kenlm_model_filename}. alpha: {kenlm_alpha}, beta {kenlm_beta}")

    def _prepare_lm(self, kenlm_model_filename = '', alpha= 0, beta = 0):
        logging.debug(f"Prepare LM: {kenlm_model_filename}")
        vocab_list = list(self.asr_processor.tokenizer.get_vocab().keys())
        with open("/data/voshpde/pipeLine/vocab_list.txt", 'wt') as vl:
            vl.write(str(vocab_list) + '\n')
            
            vl.write(str(len(vocab_list)) + '\n')


        # convert ctc blank character representation
            # vocab_list[0] = ''
        vocab_list[0] = ''
        # replace special characters
        vocab_list[1] = ' '
        vocab_list[2] = "⁇"
        vocab_list[3] = "⁇"
        # convert space character representation
        vocab_list[4] = ' '
        # specify ctc blank char index, since conventionally it is the last entry of the logit matrix
        alphabet = Alphabet.build_alphabet(vocab_list, ctc_token_idx=0)
        with open("/data/voshpde/pipeLine/alphabet.txt", 'wt') as vl:
            vl.write(str(alphabet._labels) + '\n')
            vl.write(pformat(alphabet.__dict__))
            # vl.write(str(len(vocab_list)))

        if kenlm_model_filename:
            logging.debug(f"Build decoder with {kenlm_model_filename}")
            kenlm_model = kenlm.Model(kenlm_model_filename)
            with open("/data/voshpde/pipeLine/kenlm_model_loaded.txt", 'wt') as out:
                # out.write("valid")
                out.write(pformat(dir(kenlm_model)))
            
            self.decoder = build_ctcdecoder(
                vocab_list,
                kenlm_model,
                ctc_token_idx = 0,
                alpha=alpha,
                beta=beta
            )
            with open('/data/voshpde/pipeLine/decoder_dict.txt', 'wt') as dec:
                dec.write(pformat(self.decoder.__dict__))
            
        else:    
        # build the decoder and decode the logits
            self.decoder = BeamSearchDecoderCTC(alphabet)
        return self.decoder
        

    def _get_text(self, filename:str, hotwords=[], hotwords_weight = 0):

        try:
            speech, rate = librosa.load(filename, sr=16000)

        
            input_values = self.asr_processor(speech, return_tensors="pt", sampling_rate=rate).input_values  
            # logits = self.asr_model(input_values).logits #.cpu().detach().numpy()[0]

            use_pyctcdecode = True
            if use_pyctcdecode:
                logits = self.asr_model(input_values).logits.cpu().detach().numpy()[0]
                with open("logits.txt", 'wt') as logits_out:
                    logits_out.write(str(logits.shape))

                if hotwords:
                    transcription = self.decoder.decode(logits, hotwords=hotwords, hotword_weight=hotwords_weight)
                else:
                    transcription = self.decoder.decode(logits)
            else:
                #Store predicted id's
                logits = self.asr_model(input_values).logits #.cpu().detach().numpy()[0]
                predicted_ids = torch.argmax(logits, dim =-1)
                #decode the audio to generate text
                transcription = self.asr_tokenizer.decode(predicted_ids[0])
            return transcription
        except ValueError as e:
            logging.warning(e)
            raise ValueError(e)
            return ''

    def transcribe_wavfile(self, filename:str):
        if not os.path.exists(filename):
            raise RuntimeError(f"Cannot transcribe {filename}. File does not exist.")

        text = self._get_text(filename)
        return text


    # print(transcriptions)

    def results_to_json(self, results:InferenceResults, filename:str):
        if not filename.endswith(".json"):
            raise ValueError(f"Filename must have the .json extension")

        with open(filename, 'wt') as out:
            json.dump(asdict(results), out)

        logging.info(f"Wrote json to {filename}")


    def results_to_csv(self, results:list, filename:str):
        if not filename.endswith(".csv"):
            raise ValueError(f"Filename must have the .csv extension")

        with open(filename, 'wt') as out:
            mywriter = writer(out, delimiter = '\t')
            mywriter.writerow(['inferred_text', 'duration' , 'start_seconds', 'end_seconds', 'segment_wavfile', 'original_wavfile'])
            for result in results.segments:
                print([result.inferred_text, result.end_seconds-result.start_seconds , result.start_seconds, result.end_seconds, result.wavfile, results.original_wavfile])
                mywriter.writerow([result.inferred_text, result.end_seconds-result.start_seconds , result.start_seconds, result.end_seconds, result.wavfile, results.original_wavfile])

        logging.info(f"Wrote csv to {filename}")

    def _make_json_filename(self, results:InferenceResults, folder = './') -> str:
        basename = os.path.basename(results.original_wavfile)
        naked_basename = os.path.splitext(basename)[0]
        json_filename = f"{naked_basename}.json"
        json_filename = os.path.join(folder, json_filename)
        return json_filename

    def _make_csv_filename(self, results:InferenceResults, folder = './') -> str:
        basename = os.path.basename(results.original_wavfile)
        naked_basename = os.path.splitext(basename)[0]
        csv_filename = f"{naked_basename}.csv"
        csv_filename = os.path.join(folder, csv_filename)
        return csv_filename


    def TranscribeFromBookkeep(self, segmentationbookkeep:LengthSegmentationBookkeep, language_model='', json_folder='./json_output2', csv_folder='./csv_output2'):
        logging.debug("THE MODELNAME TRANSCRIBE FROM BOOKKEEP", self.model_name)
        if not os.path.exists(json_folder):
            os.makedirs(json_folder)
            logging.info(f"Created folder: {json_folder}")
        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)
            logging.info(f"Created folder: {csv_folder}")
        
        results = InferenceResults(
            original_wavfile = segmentationbookkeep[0]["original_filename"],
            segments = []
        )

        for segment in segmentationbookkeep:
            text = self.transcribe_wavfile(segment["filename"])
            inference_result = InferenceResultSegment(
                wavfile = segment["filename"],
                
                start_seconds = segment["start_seconds"],
                start_frames = segment["start_frames"],
                
                end_seconds = segment["end_seconds"],
                end_frames = segment["end_frames"],

                # diarization_turn_i = segment.diarization_turn_i,
                identifier = segment["identifier"],

                # speaker = segment.speaker,
                inferred_text = text

            )
            results.segments.append(inference_result)

        json_filename = self._make_json_filename(results, json_folder)
        logging.info(f"json filename {json_filename}")
        csv_filename = self._make_csv_filename(results, csv_folder)
        logging.info(f"csv filename {csv_filename}")
        self.results_to_json(results, json_filename)
        self.results_to_csv(results, csv_filename)
