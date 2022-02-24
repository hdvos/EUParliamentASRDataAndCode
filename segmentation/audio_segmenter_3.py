# from segmentation.Segmentation import segment_sound_wave_from_bookkeep
import numpy as np
import math
from scipy.io import wavfile
import os
import pandas as pd
from pprint import pprint
from dataclasses import dataclass
import logging
from collections import namedtuple

DEBUG = False

WavdataContainer = namedtuple("WavdataContainer", ["data", "rate"])
WAV_FILES = "/data/voshpde/wav_files/zipfiles"
CSV_FOLDER = "csv_files3"
if not os.path.exists(CSV_FOLDER):
    os.makedirs(CSV_FOLDER)

@dataclass
class Segment:
    wavdata:WavdataContainer
    start_seconds:float
    end_seconds:float

    start_frames:int
    end_frames:int

def read_wavfile(filename):
    logging.debug(f"read {filename}")
    rate, data = wavfile.read(filename)
    
    logging.debug(f"loaded {filename}")
    logging.debug(f"Rate: {rate}")

    wavdata = WavdataContainer(data, rate)
    
    naked_filename = os.path.basename(filename)
    filesize_GB = os.path.getsize(filename)/(1024**3)
    logging.debug(f"{naked_filename} ({filesize_GB:.2f} GB) loaded.")
    logging.debug("Shape: " + str(wavdata.data.shape))
    
    
    return wavdata


def is_normalized(data:np.ndarray) -> bool:
    """Checks if a waveform is already normalized.

    :param data: a waveform
    :type data: np.ndarray
    :return: True if data is normalized, else, false.
    :rtype: bool
    """
    return 0.8 < data.max() <= 1 or -0.8 > data.min() >= -1
        

def normalize_wave(data:np.ndarray, bits_per_sample:int = 16) -> np.ndarray:
    """Normalizes a wave signal such that all values are between -1 and 1

    :param data: a single channel wave signal.
    :type data: np.ndarray
    :param bits_per_sample: The number of bits per sample, defaults to 16
    :type bits_per_sample: int, optional
    :return: normalized waveform
    :rtype: np.ndarray
    """
    assert not is_normalized(data), "Data is already normalized"
    
#     if self.normalized:
#         raise RuntimeError("Data was already normalized")
    print(f"---Before normalization: min: {data.min()} - max: {data.max()}")
    # data_normalized = data / 2**(bits_per_sample-1)
    data_normalized = data/ ( max(abs(data.min()), data.max()) +1)
    print(f"---After normalization: min: {data_normalized.min()} - max: {data_normalized.max()}")
    assert is_normalized(data_normalized), "Data is not properly normalzed, try using another value for bits_per_sample"

    return data_normalized

def channels_similar(data:np.ndarray, r_threshold:float = .95, p_threshold:float = .005) -> bool:
    """Checks if two channels are similar enough in order for one to be removed. The two channels are similar if pearsons R is larger than the given threshold.

    :param data: A wave signal.
    :type data: np.ndarray
    :param r_threshold: The minimun value for R to determine whether two channels are similar, defaults to .95
    :type r_threshold: float, optional
    :param p_threshold: The maximum value for p below which the pearson R is significant, defaults to .005
    :type p_threshold: float, optional
    :return: True if channels are similar (enough) and false if channels are (too) different.
    :rtype: bool
    """
    assert data.shape[1] == 2, "The data should have two channels in order to compare them."
    print("--- Check if channels are similar...")
    # r, p = pearsonr(data[:,0], data[:,1])
    # return r > r_threshold and p < p_threshold
    return True

def remove_channel_if_similar(data:np.ndarray) -> np.ndarray:
    """Removes 1 channel of the wav-signal, except when they are different.

    :param data: wav signal with two channels
    :type data: np.ndarray
    :raises ValueError: ValueError if chanels are not similar. #TODO 1: find a more appropriate exception. #TODO 2: different strategy for when channels are dissimilar.
    :return: [description]
    :rtype: np.ndarray
    """
    assert data.shape[1] == 2, "The data should have two channels in order to remove one."

    if channels_similar(data):
        print ("--- Channels are similar")
        return data[:,0]
    else:
        raise ValueError("Data channels are not similar, so not removing one.")

def prepare_data_for_segmentation(wavdata, filenm):
    data=wavdata.data

    print("-- Normalize data ..")
    data_norm = normalize_wave(data)
    print(f"-- Remove channel {data_norm.shape} ..")
    try:
        data_norm_single = remove_channel_if_similar(data_norm)
    except IndexError as msg:
        print("xxxx IndexError")
        data_norm_single = data_norm
        with open("IndexErrorLog.txt", 'a') as o:
            o.write(f"{filenm}\t{msg}\n{'-'*50}")
    except AssertionError as msg:
        data_norm_single = data_norm
        print("xxxx ASSERTIONERROR")
        with open("AssertionErrorLog.txt", 'a') as o:
            o.write(f"{filenm}\t{msg}\n{'-'*50}")


    print("-- Data prepared!")
    prepared_wavdata = WavdataContainer(np.array(data_norm_single), wavdata.rate)
    return prepared_wavdata


#%%

def get_wavfile_core(wavfile_name:str) -> str:
    """Get the name of a (wav) file without the path before and the trailing file extension. E.g.: foo/bar/baz.wav -> baz
    NOTE: this could also have been done with os.path methods. TODO: rewrite with os.path methods.

    :param wavfile_name: name of a wav-file
    :type wavfile_name: str
    :return: the filename without the path and trailing file extension.
    :rtype: str
    """

    # wavfile_core = wavfile_name.split("/")[-1].split('.')[0]
    basename = os.path.basename(wavfile_name)
    wavfile_core  = os.path.splitext(basename)[0]
    # print(f"---Wavfile core = {wavfile_core}")
    return wavfile_core


def make_wavfile_output_folder(original_wavfile_name:str, location:str = "separated_wavfiles3") -> None:
    """Makes a folder where segments will be stored.

    :param original_wavfile_name: The filename of the original wavfile
    :type original_wavfile_name: str
    :param location: the location where the folder needs to be created, defaults to "separated_wavfiles"
    :type location: str, optional
    :raises RuntimeError: If the program has no permission to create the foler.
    """

    if not os.path.exists(location):
        os.mkdir(location)
        
    wavfile_core = get_wavfile_core(original_wavfile_name)
    wavfiles_location = os.path.join(location, wavfile_core)
    if not os.path.exists(wavfiles_location):
        os.makedirs(wavfiles_location)
    else:
        print(f"--- Clear folder")
#         input(f"rm {wavfiles_location}/*")
        os.system(f"rm {wavfiles_location}/*")
    
    return wavfiles_location

def make_filepath(folder, base, index):
    filename = f"{index:03d}_{base}.wav"
    return os.path.join(folder, filename)
    

def write_wav(segment, fragment_index, folder, filename_base = 'test'):
    filepath = make_filepath(folder, filename_base, fragment_index)
    print(f"--- Write {filepath}")
    wavfile.write(filepath, segment.wavdata.rate, segment.wavdata.data)
    filepath_pcm = filepath + ".pcm"
#     print("--- Convert to pcm.")
    print(f"---- Transform to {filepath_pcm}")
    os.system( f"sox {filepath} -t wav -r 16000 -b 16 {filepath_pcm}")
    print(f"---- Remove {filepath}")
    os.system( f"rm {filepath}")
    assert os.path.exists(filepath_pcm), "filepath should still exist"
    assert not os.path.exists(filepath), "filepath should be removed"
    
    os.system( f"mv {filepath_pcm} {filepath}")

    assert not os.path.exists(filepath_pcm), "filepath should still exist"
    assert os.path.exists(filepath), "filepath should be removed"

    # print("-"*30)

#     print(f"--- Converted to pcm. \n--- New filename: {filepath_pcm}")
    
    return filepath

#%% 

KernelData = namedtuple("KernelData", ["left", "center", "right", "energy", "data"])

def calculate_signal_energy(kernel_signal:np.array) -> float:
    """Calculate the average energy of a sound wave. Energy: mean squared amplitude

    :param kernel_signal: A numpy array
    :type kernel_signal: np.array
    :return: The energy of a segment (kernel) of wav.
    :rtype: float
    """
    assert len(kernel_signal) != 0, "Signal cannot be empty"
    return (kernel_signal**2).sum() / len(kernel_signal)

def search_window_for_breaking_point(wavdata_onechannel:np.array, search_window_min:float, search_window_max:float, stride:float=0.1, kernel_width:float = 0.5) -> KernelData:
    """Search within a window for the most ideal time to break of the wav signal. Most ideal: most silent point.

    NOTE: this script was written for data recorded in mono. It will have to be adapted if data is recorded in stereo.
    :param wavdata_onechannel: A single channel of the wav signal.
    :type wavdata_onechannel: np.array
    :param search_window_min: in seconds: the start of the search window (The window where you look for a breaking point)
    :type search_window_min: float
    :param search_window_max: in seconds: the end of the search window.
    :type search_window_max: float
    :param stride: the step size of the kernel shift, defaults to 0.1
    :type stride: float, optional
    :param kernel_width: the width of the search kernel, defaults to 0.5
    :type kernel_width: float, optional
    :return: return Kerneldata (a named tuple) with data about the most optimal breaking point 
    :rtype: KernelData
    """
    
    assert search_window_min > 0, 'search_window_min must be positive' #1
    assert search_window_max > 0, 'search_window_max must be positive' #2
    assert search_window_min < search_window_max, "Search window min must be larger than search windwo max" #4
    assert stride > 0, "stride must be a positive number" #5
    assert kernel_width < (search_window_max - search_window_min), "Kernel must fit inside the search space." #6
    assert kernel_width > 0, "Kernel width must be a positive number" #7

    print("------------------------------")
    kernel_half = kernel_width/2
    kernel_half_frames = math.floor(kernel_half*wavdata_onechannel.rate) # Convert seconds to frames

    stride_frames = math.floor(stride*wavdata_onechannel.rate)  # Convert stride width to frames

    # Define the kernel
    kernel_center = search_window_min + kernel_half_frames 
    kernel_left = kernel_center - kernel_half_frames
    kernel_right = kernel_center + kernel_half_frames

    # Kernel exceeds search space. NOTE: Probably redundant with assertion 6
    if kernel_right > search_window_max:
        return None

    best_kernel = KernelData(kernel_left, kernel_center, kernel_right, 1, wavdata_onechannel.data[kernel_left:kernel_right])

    # Search for the best kernel
    while True:
        if len(wavdata_onechannel.data[kernel_left : kernel_right]) == 0:
            break #TODO: this is a stupid hack. Solve.
        kernel_energy = calculate_signal_energy(wavdata_onechannel.data[kernel_left : kernel_right])

        # Check if current kernel is better than best kernel. Better: more silent (lower energy).
        if kernel_energy < best_kernel.energy:
            best_kernel = KernelData(kernel_left, kernel_center, kernel_right, kernel_energy, wavdata_onechannel.data[kernel_left : kernel_right])
            #print(best_kernel)
            # input()

        # calculate new kernel
        kernel_center += stride_frames
        kernel_left = kernel_center - kernel_half_frames
        kernel_right = kernel_center + kernel_half_frames


        if kernel_right > search_window_max:
            break

    
    # print("Best Kernel:", best_kernel)
    return best_kernel

def segment_sound_wave(wavdata_onechannel:np.array, desired_segment_length:float = 10.0, min_segment_length:float = 5.0, max_segment_length:float = 30.0, 
                        stride:float = 0.1, kernel_width:float = 0.5) -> list:
    assert desired_segment_length > 0, "Must be positive"   #1
    assert min_segment_length >= 0, "segment length must be positive"   #2
    assert min_segment_length <= desired_segment_length, "Min segment length must be smaller or equal than desired segment length" #3
    assert max_segment_length >= desired_segment_length, "max segment length must be larger or equal than desired segment length" #4
    
    segments = []

    time_step = stride * wavdata_onechannel.rate    # seconds to frames
    current_location = 0

    window_minus = (min_segment_length - desired_segment_length) * wavdata_onechannel.rate
    window_plus  = (max_segment_length - desired_segment_length) * wavdata_onechannel.rate
    
    # Loop setup
    steps = 0    
    search_window_center = current_location + time_step
    search_window_min = search_window_center - window_minus
    search_window_max = search_window_center + window_plus
    search_window_max = min(search_window_max, len(wavdata_onechannel.data))
    
    while True:
        
        breaking_point = search_window_for_breaking_point(wavdata_onechannel, search_window_min, search_window_max)
        if not breaking_point:
            segment_wavdata = WavdataContainer(wavdata_onechannel.data[current_location:], wavdata_onechannel.rate)
            segment = Segment(wavdata = segment_wavdata, 
                              start_seconds = current_location/segment_wavdata.rate,
                              end_seconds=len(wavdata_onechannel.data)/segment_wavdata.rate,
                              start_frames= current_location,
                              end_frames=len(wavdata_onechannel.data))
            segments.append(segment)
            break
        
        segment_wavdata = WavdataContainer(wavdata_onechannel.data[current_location:breaking_point.center], wavdata_onechannel.rate)
        segment = Segment(wavdata = segment_wavdata, 
                          start_seconds= current_location/segment_wavdata.rate,
                          end_seconds=breaking_point.center/segment_wavdata.rate,
                          start_frames= current_location,
                          end_frames=breaking_point.center)
        segments.append(segment)
                
        # Update
        current_location = breaking_point.center + 1
        
        search_window_center = current_location + time_step
        search_window_min = search_window_center - window_minus
        search_window_max = search_window_center + window_plus
        search_window_max = min(search_window_max, len(wavdata_onechannel.data))
        steps += 1
        
        if current_location >= len(wavdata_onechannel.data):
            break
        # print(data[current_location])
    # print(segments[-1])
    return segments


def process_file(filenm_gz:str):
    """Main logic for segmenting a single wavfile.

    :param filenm: Path to the original wavfile
    :type filenm: str
    :param csvfile_output_location: Location (folder) where the segments will be stored, defaults to 'gridsearch_results'
    :type csvfile_output_location: str, optional
    """
    print(filenm_gz)
    filenm = filenm_gz.replace(".gz", '')


#    os.system(f"gunzip -k {filenm_gz}")
    os.system(f"gunzip -c {filenm_gz} > {filenm}")
    assert os.path.exists(filenm_gz), "Gzip file should still exist"
    assert os.path.exists(filenm), "Unzipped file must be created"
    # # TODO: UNZIP

    try:
        wavdata = read_wavfile(filenm)
        print(wavdata.data.shape)
        wavdata_onechannel = prepare_data_for_segmentation(wavdata, filenm)
        print(wavdata_onechannel.data.shape)

        segments = segment_sound_wave(wavdata_onechannel)
        # pprint(segments)
        segments_metadata = []

        wavfiles_location = make_wavfile_output_folder(filenm)
        filenm_core = get_wavfile_core(filenm)

        print(wavfiles_location)
        for i, segment in enumerate(segments, 1):
            segment_filename = write_wav(segment, i, wavfiles_location, filenm_core)
            segment_length = len(segment.data)/segment.rate
            identifier = filenm_core
            segment_metadata = { "identifier":identifier,
                                 "segment_length": segment_length,
                                 "filename":segment_filename,
                                  }
            segments_metadata.append(segment_metadata)

        segments_metadata_df = pd.DataFrame.from_dict(segments_metadata)
        csv_filename = os.path.join(CSV_FOLDER, f"{filenm_core}_metadata.csv")
        segments_metadata_df.to_csv(csv_filename, sep="\t")

        file_metadata = {   "filename":os.path.basename(filenm_gz),
                            "average_length": segments_metadata_df.segment_length.mean(),
                            "nr_segments":segments_metadata_df.shape[0]
                        }

        

    finally:
        os.system(f"rm {filenm}")
        assert os.path.exists(filenm_gz), "Gzip file should still exist"
        assert not os.path.exists(filenm), "Unzipped file must be deleted"

    return file_metadata


def segment_wavfile(wavfilename:str, wavfile_folder:str):
    """Main logic for segmenting a single wavfile.

    :param filenm: Path to the original wavfile
    :type filenm: str
    :param csvfile_output_location: Location (folder) where the segments will be stored, defaults to 'gridsearch_results'
    :type csvfile_output_location: str, optional
    """
    logging.debug(f"Segmenting {wavfilename}")

    try:
        wavdata = read_wavfile(wavfilename)
        logging.debug(wavdata.data.shape)
        wavdata_onechannel = prepare_data_for_segmentation(wavdata, wavfilename)
        
        segments = segment_sound_wave(wavdata_onechannel)
        logging.info(f"Wavfile segmented. Nr segments: {len(segments)}")

        segments_metadata = []
        
        wavfiles_location = make_wavfile_output_folder(wavfilename, wavfile_folder)
        
        logging.info(f"Segmented files in folder: {wavfiles_location}")

        filenm_core = get_wavfile_core(wavfilename)
        
        for i, segment in enumerate(segments, 1):
            segment_filename = write_wav(segment, i, wavfiles_location, filenm_core)
            
            segment_length = len(segment.wavdata.data)/segment.wavdata.rate
            identifier = filenm_core
            segment_metadata = { "identifier":identifier,
                                 "segment_length": segment_length,
                                 "filename":segment_filename,
                                 "original_filename":wavfilename,
                                 "start_seconds": segment.start_seconds,
                                 "end_seconds": segment.end_seconds,
                                 "start_frames": segment.start_frames,
                                 "end_frames":segment.end_frames
                                  }
            
            segments_metadata.append(segment_metadata)

        segments_metadata_df = pd.DataFrame.from_dict(segments_metadata)
        segments_metadata_df['length_s'] = segments_metadata_df.end_seconds - segments_metadata_df.start_seconds
        csv_filename = os.path.join(CSV_FOLDER, f"{filenm_core}_metadata.csv")
        
        segments_metadata_df.to_csv(csv_filename, sep="\t")
        logging.info(f"segmentation metadata csv written to {csv_filename}")

        file_metadata = {   "filename":os.path.basename(wavfilename),
                            "average_length": segments_metadata_df.segment_length.mean(),
                            "nr_segments":segments_metadata_df.shape[0]
                        }

        

    finally:
        ...

    return file_metadata, segments_metadata









# def segment_all_files(folder="wav_files"):
#     assert os.path.exists(folder)
#     filenames = os.listdir(folder)
#     filenames = [os.path.join(folder, filenm) for filenm in filenames if filenm.endswith(".gz")]

#     files_metadata = []

#     for filenm in filenames:
#         try:
#             file_metadata = process_file(filenm)
#             files_metadata.append(file_metadata)
#             print(file_metadata)
#         except IndexError as msg:
#             with open("AssertionErrorLog.txt", 'a') as o:
#                 o.write(f"{filenm}\t{msg}\n{'-'*50}")

#     all_metadata = pd.DataFrame.from_dict(files_metadata)
#     all_metadata.to_csv("audio_segmenter_2_results.csv", sep='\t')

if __name__ == "__main__":
    # segment_all_files(WAV_FILES)
    segment_wavfile('../testfile2.wav')
    # file_metadata = process_file(testfilename)
    # print(file_metadata)
