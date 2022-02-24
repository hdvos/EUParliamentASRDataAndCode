import numpy as np
import math
from scipy.io import wavfile
import os
import pandas as pd
from pprint import pprint
from diarization.Diarization import DiarizationBookkeep, DiarizationBookkeepSegment, MyTurnContainer
from dataclasses import dataclass, asdict
from collections import namedtuple
import shutil
import json


WavdataContainer = namedtuple("WavdataContainer", ["data", "rate"])
# WAV_FILES = "/data/voshpde/wav_files/zipfiles"
# CSV_FOLDER = "csv_files3"
# if not os.path.exists(CSV_FOLDER):
    # os.makedirs(CSV_FOLDER)

def read_wavfile(filename):
    rate, data = wavfile.read(filename)
    wavdata = WavdataContainer(data, rate)
    # input(data)
    naked_filename = os.path.basename(filename)
    filesize_GB = os.path.getsize(filename)/(1024**3)
    print(f"{naked_filename} ({filesize_GB:.2f} GB) loaded.")
    print("Shape: " + str(wavdata.data.shape))
    
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



# TODO: document and sort out logging
def prepare_data_for_segmentation(wavdata, filenm):
    data=wavdata.data

    print("-- Normalize data ..")
    data_norm = normalize_wave(data)
    print(f"-- Remove channel {data_norm.shape} ..")
    try:
        data_norm_single = remove_channel_if_similar(data_norm)
    except IndexError as msg:
        data_norm_single = data_norm
        with open("AssertionErrorLog.txt", 'a') as o:
            o.write(f"{filenm}\t{msg}\n{'-'*50}")
    except AssertionError as msg:
        data_norm_singe = data_norm
        with open("AssertionErrorLog.txt", 'a') as o:
            o.write(f"{filenm}\t{msg}\n{'-'*50}")


    print("-- Data prepared!")
    prepared_wavdata = WavdataContainer(np.array(data_norm_single), wavdata.rate)
    return prepared_wavdata


#%%




#%% 

KernelData = namedtuple("KernelData", ["left", "center", "right", "energy", "data"])

def calculate_kernel_energy(kernel_signal):
    assert len(kernel_signal) != 0, "Must be positive"
    return (kernel_signal**2).sum() / len(kernel_signal)

def search_window_for_breaking_point(wavdata_onechannel, search_window_min, search_window_max, stride=0.1, kernel_width = 0.5):
    print("------------------------------")
    kernel_half = kernel_width/2
    kernel_half_frames = math.floor(kernel_half*wavdata_onechannel.rate)

    stride_frames = math.floor(stride*wavdata_onechannel.rate)

    kernel_center = search_window_min + kernel_half_frames 
    kernel_left = kernel_center - kernel_half_frames
    kernel_right = kernel_center + kernel_half_frames

    if kernel_right > search_window_max:
        return None

    best_kernel = KernelData(kernel_left, kernel_center, kernel_right, 1, wavdata_onechannel.data[kernel_left:kernel_right])

    while True:
        if len(wavdata_onechannel.data[kernel_left : kernel_right]) == 0:
            break #TODO: this is a stupid hack. Solve.
        kernel_energy = calculate_kernel_energy(wavdata_onechannel.data[kernel_left : kernel_right])

        if kernel_energy < best_kernel.energy:
            best_kernel = KernelData(kernel_left, kernel_center, kernel_right, kernel_energy, wavdata_onechannel.data[kernel_left : kernel_right])
            #print(best_kernel)
            # input()

        kernel_center += stride_frames
        kernel_left = kernel_center - kernel_half_frames
        kernel_right = kernel_center + kernel_half_frames

        if kernel_right > search_window_max:
            break

    
    print("Best Kernel:", best_kernel)
    return best_kernel


@dataclass
class LengthSegmentationBookkeep():
    original_filename:str
    # diarized_filename:str
    segments:list

@dataclass
class LengthSegmentationBookkeepSegment():
    absolute_start_seconds:float
    absolute_start_frames:int
    
    absolute_end_seconds:float
    absolute_end_frames:int

    relative_start_seconds:float
    relative_start_frames:int
    
    relative_end_seconds:float
    relative_end_frames:int

    time_segmented_filename:str
    time_segment_i:int
    
    diarization_wavfile:str
    diarization_turn_i:int
    speaker:str

    time_segmented:bool

    last_segment:bool = False

def toframes(seconds:float, rate:int=44100, rounding:str='floor') -> int:
    if rounding == "floor":
        return math.floor(seconds*rate)
    elif rounding == "ceil":
        return math.ceil(seconds*rate)
    elif rounding == "round":
        return round(seconds*rate)
    else:
        raise RuntimeError(f"No rounding method named {rounding}")

def toseconds(frames, rate:int=44100) -> float:
    return frames/rate

def parse_diarization_filename(filename:str) -> list:
    basename:str = os.path.basename(filename)
    naked_basename:str = os.path.splitext(basename)[0]
    return naked_basename.split("_")

def create_wavfilename(diarization_segment:DiarizationBookkeepSegment, time_segment_i:int, root_folder:str):
    assert os.path.exists(root_folder)
    assert os.path.isdir(root_folder)
    assert type(time_segment_i) is int
    diarization_turn, speaker, identifier = parse_diarization_filename(diarization_segment.filename)
    output_folder = os.path.join(root_folder, identifier)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filename = f"{diarization_turn}_{speaker}_{time_segment_i:03d}_{identifier}.wav"
    return os.path.join(output_folder, filename)

def write_wavfile(filename:str, segment:WavdataContainer):
    print(f"--- Write {filename}")
    wavfile.write(filename, segment.rate, segment.data)
    filepath_pcm = filename + ".pcm"
#     print("--- Convert to pcm.")
    print(f"---- Transform to {filepath_pcm}")
    os.system( f"sox {filename} -t wav -r 16000 -b 16 {filepath_pcm}")
    print(f"---- Remove {filename}")
    os.system( f"rm {filename}")
    assert os.path.exists(filepath_pcm), "filepath should still exist"
    assert not os.path.exists(filename), "filepath should be removed"
    shutil.move(filepath_pcm, filename)
    assert os.path.exists(filename), "filepath should still exist"
    assert not os.path.exists(filepath_pcm), "filepath should be removed"
    print("-"*30)

def segment_sound_wave_from_bookkeep(diarization_segment:DiarizationBookkeepSegment, time_segment_bookkeep:LengthSegmentationBookkeep, wavdata_onechannel:WavdataContainer, output_root:str, desired_segment_length = 10, min_segment_length = 5, max_segment_length = 30, stride = 0.1, kernel_width = 0.5):
    assert desired_segment_length > 0, "Must be positive"
    assert min_segment_length >= 0, "segment length must be positive"
    assert min_segment_length <= desired_segment_length, "Min segment length must be smaller or equal than desired segment length"
    assert max_segment_length >= desired_segment_length, "max segment length must be larger or equal than desired segment length"
    
    segments = []


    time_step = desired_segment_length * wavdata_onechannel.rate
    current_location = 0
    current_location_seconds = math.floor(current_location/wavdata_onechannel.rate)

    time_segment_i = 1

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
        if breaking_point:
            relative_breakpoint_frames  = breaking_point.center
            relative_breakpoint_seconds = toseconds(relative_breakpoint_frames)
            
            absolute_breakpoint_seconds = diarization_segment.start + relative_breakpoint_seconds
            absolute_breakpoint_frames = toframes(absolute_breakpoint_seconds, rounding='ceil')

            relative_start_point_frames = current_location
            relative_start_point_seconds = toseconds(relative_start_point_frames)

            absolute_start_point_seconds = diarization_segment.start + relative_start_point_seconds
            absolute_start_point_frames = toframes(absolute_start_point_seconds, rounding='floor')

            if absolute_breakpoint_seconds > diarization_segment.end:
                raise RuntimeError(f"Absolute breaking point ({absolute_breakpoint_seconds}) cannot be larger than end point of diarization segment ({diarization_segment.end})")
            print(diarization_segment)
            print('breakpt', relative_breakpoint_seconds, absolute_breakpoint_seconds)
            print('breakpt', relative_breakpoint_frames, absolute_breakpoint_frames)
        if not breaking_point:
            segment = WavdataContainer(wavdata_onechannel.data[current_location:], wavdata_onechannel.rate)
            segments.append(segment)
            # create wavfilename
            # write wavfile
            absolute_start_point_seconds = diarization_segment.start + relative_start_point_seconds
            absolute_start_point_frames = toframes(absolute_start_point_seconds, rounding='floor')
            
            absolute_breakpoint_frames = len(wavdata_onechannel.data)
            absolute_breakpoint_seconds = toseconds(absolute_breakpoint_frames)

            relative_start_point_frames = current_location
            relative_start_point_seconds = toseconds(relative_start_point_frames)

            absolute_start_point_seconds = diarization_segment.start + relative_start_point_seconds
            absolute_start_point_frames = toframes(absolute_start_point_seconds, rounding='floor')

            wavfilename_out = create_wavfilename(diarization_segment, time_segment_i, output_root)
            # input(wavfilename_out)
            write_wavfile(wavfilename_out, segment)
            # input("wavfile written")

            bookkeepsegment = LengthSegmentationBookkeepSegment(
                time_segmented_filename= wavfilename_out, 
                time_segment_i = time_segment_i, 
                diarization_wavfile= diarization_segment.filename, 
                diarization_turn_i = diarization_segment.turn_i, 
                speaker = diarization_segment.speaker,

                absolute_start_seconds=absolute_start_point_seconds,
                absolute_start_frames=absolute_start_point_frames,
                
                absolute_end_seconds=absolute_breakpoint_seconds,
                absolute_end_frames=absolute_breakpoint_frames,

                relative_start_seconds=relative_start_point_seconds,
                relative_start_frames=relative_start_point_frames,

                relative_end_seconds=relative_breakpoint_seconds,
                relative_end_frames=relative_breakpoint_frames,
                
                time_segmented = True,
                last_segment=True                        
            )
            time_segment_bookkeep.segments.append(bookkeepsegment)
            break
        
        segment = WavdataContainer(wavdata_onechannel.data[current_location:breaking_point.center], wavdata_onechannel.rate)
        segments.append(segment)

        wavfilename_out = create_wavfilename(diarization_segment, time_segment_i, output_root)
        # input(wavfilename_out)
        write_wavfile(wavfilename_out, segment)
        # input("wavfile written")
        bookkeepsegment = LengthSegmentationBookkeepSegment(
            time_segmented_filename= wavfilename_out, 
            time_segment_i = time_segment_i, 
            diarization_wavfile= diarization_segment.filename, 
            diarization_turn_i = diarization_segment.turn_i, 
            speaker = diarization_segment.speaker,

            absolute_start_seconds=absolute_start_point_seconds,
            absolute_start_frames=absolute_start_point_frames,
            
            absolute_end_seconds=absolute_breakpoint_seconds,
            absolute_end_frames=absolute_breakpoint_frames,

            relative_start_seconds=relative_start_point_seconds,
            relative_start_frames=relative_start_point_frames,

            relative_end_seconds=relative_breakpoint_seconds,
            relative_end_frames=relative_breakpoint_frames,

            time_segmented = True
                        
        )

        time_segment_bookkeep.segments.append(bookkeepsegment)
        # pprint(asdict(bookkeepsegment))
        # input()
                
        # Update
        current_location = breaking_point.center + 1
        current_location_seconds = math.floor(current_location/wavdata_onechannel.rate)

        search_window_center = current_location + time_step
        search_window_min = search_window_center - window_minus
        search_window_max = search_window_center + window_plus
        search_window_max = min(search_window_max, len(wavdata_onechannel.data))
        steps += 1
        time_segment_i += 1
        
        if current_location >= len(wavdata_onechannel.data):
            break
        # print(data[current_location])
    print(segments[-1])
    return segments

def Segmentation_With_Bookkeep(diarization_segment: DiarizationBookkeepSegment, time_segment_bookkeep:LengthSegmentationBookkeep, output_root:str):
    """Segment an already diarized wavfile in segments between 10 and 30 seconds.

    :param diarization_segment: A bookkeeping segment from the diarization bookkeeping.
    :type diarization_segment: DiarizationBookkeepSegment
    :param time_segment_bookkeep: a bookkeeping object where the bookkeping of this segmentation will be done/
    :type time_segment_bookkeep: LengthSegmentationBookkeep
    :param output_root: The folder where the resulting wavfiles will be stored.
    :type output_root: str
    """
    wavdata = read_wavfile(diarization_segment.filename)
    wavdata_onechannel = prepare_data_for_segmentation(wavdata, diarization_segment.filename)
    print(wavdata_onechannel.data.shape)
    wav_segments = segment_sound_wave_from_bookkeep(diarization_segment, time_segment_bookkeep, wavdata_onechannel, output_root)


def store_segmentation_bookkeep(time_segment_bookkeep:LengthSegmentationBookkeep, filename:str):
    """Store the bookkeeping as a json file in the desired place.

    :param time_segment_bookkeep: The bookeeping object
    :type time_segment_bookkeep: LengthSegmentationBookkeep
    :param filename: The filename.
    :type filename: str
    :raises ValueError: If the filename does not end in .json
    """
    if not filename.endswith(".json"):
        raise ValueError(f"{filename} is invalid. Please enter a json filename (with the .json extension)")

    with open(filename, 'wt') as out:
        json.dump(asdict(time_segment_bookkeep), out)

    print(f"Json stored at {filename}")

def SegmentTurnsFromBookkeep(bookkeepdata:DiarizationBookkeep, output_root:str, bookkeep_json_file:str ,segmentation_worthyness:float = 45.0) -> LengthSegmentationBookkeep:
    """Takes a diarization bookkeep object and segment wavfile unless they are too short to be segmented.

    :param bookkeepdata: An object containing bookkeeping data as a result of diarization.
    :type bookkeepdata: DiarizationBookkeep
    :param output_root: The root folder where the output folder  structure will be created. Every "Motherfile" will get its own folder containing the segmented files.
    :type output_root: str
    :param bookkeep_json_file: the name of the json file that will be the result of this
    :type bookkeep_json_file: str
    :param segmentation_worthyness: if a wavfile is shorter than this, it will not be further segmented, defaults to 45.0
    :type segmentation_worthyness: float, optional
    """
    assert os.path.exists(output_root), "Output folder should exist"
    assert os.path.isdir(output_root), "Output root should be a directory"

    # Initialize the bookkeeping object.
    time_segment_bookkeep = LengthSegmentationBookkeep(
        original_filename=bookkeepdata.original_file,
        # diarized_filename='',
        segments=[]
    )

    for diarization_segment in bookkeepdata.segments:
        # If the segment is too short, it will be copied as whole and added to the bookkeeping object.
        if (diarization_segment.end - diarization_segment.start) < segmentation_worthyness:
            
            wavfilename_out:str = create_wavfilename(diarization_segment, 0, output_root)
            shutil.copy2(diarization_segment.filename, wavfilename_out)
            
            bookkeepsegment = LengthSegmentationBookkeepSegment(
                time_segmented_filename= wavfilename_out, 
                time_segment_i = 0,     # If it is copied as a whole, the time segment i is set to 0.
                diarization_wavfile= diarization_segment.filename, 
                diarization_turn_i = diarization_segment.turn_i, 
                speaker = diarization_segment.speaker,

                absolute_start_seconds=diarization_segment.start,
                absolute_start_frames=toframes(diarization_segment.start),
                
                absolute_end_seconds=diarization_segment.end,
                absolute_end_frames=toframes(diarization_segment.end),

                relative_start_seconds=0,
                relative_start_frames=0,

                relative_end_seconds=diarization_segment.end - diarization_segment.start,
                relative_end_frames=toframes(diarization_segment.end - diarization_segment.start, rounding='floor'),
                
                time_segmented = False, # It is not segmented based on length.
                last_segment=True       # If it is the only segment, it is therefore the last.                 
            )
            time_segment_bookkeep.segments.append(bookkeepsegment)

        else:
            Segmentation_With_Bookkeep(diarization_segment, time_segment_bookkeep, output_root)

        
    # pprint(asdict(time_segment_bookkeep))
    store_segmentation_bookkeep(time_segment_bookkeep, bookkeep_json_file)
    return time_segment_bookkeep

# #TODO remove main?
# if __name__ == "__main__":
#     segment_all_files(WAV_FILES)
#     # file_metadata = process_file(testfilename)
#     # print(file_metadata)

# %%
