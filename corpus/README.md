# Corpus

This folder contains the automatically transcribed files.

Every file is named after the audio file that it transcribes. The filename represents the dat eof the meeting. Due to the size (about 300GB) it is not possible to publish all original wav files. If you want them, contact me, then we can see what is possible

## Columns

Every tsv file contains the following columns. 

* wavfile: refers to the exact wavfile that was transcribed. (The the big wav files were split in smaller wav files of about 20 sconds for efficiency.)
* start_seconds: the start in seconds of the smaller wavfiles after the start of the main wav file.
* start_frames: the number of frames from the beginning of the main wav file.	
* end_seconds: the end in seconds of the smaller wavfile after the start of the main wav file.
* end_frames: the end in frames of the smaller wavfiles after the start of the main wav file.	
* identifier: a unique identifier for this piece of text.
* inferred_text_with_hotwords: the automatically transcribed text with using a hotwords list.
* inferred_text_without_hotwords: the automatically transcribed without using a hotword list
* motherfile: the main wavfile the sub-wavfile is a part of.
* dist: the levenshtein distance between the text with hotwords and the text without hotwords.



