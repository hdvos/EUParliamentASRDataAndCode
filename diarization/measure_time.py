# load pipeline
import torch
import time
import os
import shutil
import sox

def prepare_output_folder(folder_location):
    if not os.path.exists(folder_location):
        os.makedirs(folder_location)
    else:
        shutil.rmtree(folder_location)
        os.makedirs(folder_location)    
    assert os.path.exists(folder_location), 'If folder is not created, sth is seriously wrong.'


WAVFILES_LOCATION = '/data/voshpde/wav_files/zipfiles'
# RTTM_OUT = 'test_rttm.rttm'
TIMELOG_FILE = 'timelog.txt'
RTTM_OUTPUT_FOLDER = 'rttm_files'
READABLE_OUTPUT_FOLDER = 'readable_files'
TMPDIR = 'tmp'
PROCESS_N =20

# sox.file_info.duration(unzipped_name)
if __name__ == "__main__":
    prepare_output_folder(RTTM_OUTPUT_FOLDER)
    prepare_output_folder(READABLE_OUTPUT_FOLDER)
    prepare_output_folder(TMPDIR)
    
    with open(TIMELOG_FILE, 'wt') as f:
        f.write('FILE DURATION\tDIARIZATION DURATIO\n')
    
    pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')
    
    wavfile_list = os.listdir(WAVFILES_LOCATION)
    wavfile_list = [os.path.join(WAVFILES_LOCATION, wavfile) for wavfile in wavfile_list]
    
    for wavfile_zip in wavfile_list[:PROCESS_N]:
        try:
            wavfile_basename = os.path.splitext( os.path.basename(wavfile_zip))[0]
            wavfilename_unzipped = os.path.join(TMPDIR, wavfile_basename)
            os.system(f"gunzip -c {wavfile_zip} > {wavfilename_unzipped}")
            assert os.path.exists(wavfilename_unzipped), "The wavfile should be unzipped and in said location"
            assert os.path.exists(wavfile_zip), "zipfile should not be removed"
            
            start_time = time.time()
                                
            # apply diarization pipeline on your audio file
            diarization = pipeline({'audio': wavfilename_unzipped})
            
            end_time = time.time()
            diarization_duration = end_time - start_time
            file_duration = sox.file_info.duration(wavfilename_unzipped)
            
            with open(TIMELOG_FILE, 'a') as out:
                out.write(f"{file_duration:.2f}\t{diarization_duration:.2f}\n")
        finally:
            os.remove(wavfilename_unzipped)
            assert not os.path.exists(wavfilename_unzipped), "The wavfile should be removed"
            assert os.path.exists(wavfile_zip), "zipfile should not be removed"
            
        # dump result to disk using RTTM format
        with open(os.path.join(RTTM_OUTPUT_FOLDER , f'{wavfile_basename}.rttm'), 'w') as f:
            diarization.write_rttm(f)
      
        # iterate over speech turns
        with open(os.path.join(READABLE_OUTPUT_FOLDER, f'{wavfile_basename}.txt'), 'wt') as out:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                out.write(f'Speaker "{speaker}" speaks between t={turn.start:.1f}s and t={turn.end:.1f}s.\n')
                
# Speaker "A" speaks between t=0.2s and t=1.4s.
# Speaker "B" speaks between t=2.3s and t=4.8s.
# Speaker "A" speaks between t=5.2s and t=8.3s.
# Speaker "C" speaks between t=8.3s and t=9.4s.
# ...
