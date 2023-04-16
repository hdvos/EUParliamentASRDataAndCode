# EU Parliament ASR Data and Code

* The **diarization** folder contains all files for the diarization experiment.

* The **segmentation** folder contains scripts for segmenting audio files in segments of 15-25 seconds.

* The **transcription** folder contains scripts for transcribing a wav-file/

* ASRpipeline.py contains the entire pipeline. To transcribe a meeting (or other wav-files) run this script.

* the **Corpus** folder contains the transcribed LIBE meetings.

## ASRpipeline.py

Run `ASRpipeline.py <path_to_wavfile>` to process a wav file.

### Parameters

* `wavfile`: the wavfile you want to process
* `-rttm_output`: path to where the rttm files need to be stored if you use diarization.
* `-segmentation_output_folder`: the folder where the results of segmentation will be stored. 
* `-segmentation_bookkeep_file`: The name of the json file with the segmentation bookkeep. This file keeps track of how the wav file is segmented.
* `-asr_model`: The name of the wav2vec model that is used for the asr. default="facebook/wav2vec2-base-10k-voxpopuli-ft-en",
* `-kenlm_model`: The name of the knlm model that is used for decoding the ASR results.
* `-transcriptions_output_folder`: "Folder where the transcriptions will be stored.
* `-logfolder`: The folder where the logfiles will go.
