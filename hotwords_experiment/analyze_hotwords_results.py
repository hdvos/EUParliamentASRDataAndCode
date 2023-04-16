import os
import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer
import jiwer

print('import spacy ... ', end='')
import spacy 
print("DONE")

from nltk.corpus import stopwords
import unidecode
from dataclasses import dataclass, asdict

print('load spacy model ... ', end='')
nlp = spacy.load("en_core_web_sm")
print('DONE')

root_folder = '/home/hugo/MEGA/work/ASR/EUParliamentASR/hotwords_experiment/results'

stops = set(stopwords.words('english'))

ENTITY_TYPES_OF_INTEREST = ["ORG", "PERSON", 'GPE', 'LOC']

PLOTS_FOLDER = "/home/hugo/MEGA/work/ASR/EUParliamentASR/hotwords_experiment/plots"

MIN_ENTITY_WORD_LEN = 2

TABLES_FOLDER = '/home/hugo/MEGA/work/ASR/EUParliamentASR/hotwords_experiment/processed_data'
if not os.path.exists(TABLES_FOLDER):
    os.makedirs(TABLES_FOLDER)

def get_hotwords_weight(filename:str) -> int:
    a = os.path.splitext(filename)[0]
    a = a.split('_')[-1]
    a = int(a)
    return a

def extract_entities_from_reference(text):
    text = str(text)
    if not text:
        # # # input(text)
        return []
    
    # print("process_text ... ", end=' ')
    doc = nlp(text)
    # print(u'\u2713')
    # # input(doc.ents)
    entities = set()

    for ent in doc.ents:
        entity = ent.text
        entity_label = ent.label_
        # print(ent.text,  repr(entity_label))

        if entity_label in ENTITY_TYPES_OF_INTEREST:
            # # input(entity)
            entity_words = entity.split(' ')
            entity_words = [word.strip().lower() for word in entity_words]
            entity_words = [word for word in entity_words if not word in stops and len(word) >= MIN_ENTITY_WORD_LEN and word.isalpha() and not word == 'ehm']
            entity_words = [unidecode.unidecode(word) for word in entity_words]
            # input(entity_words)
            entities.update(entity_words)
    # print(entities)
    # input('*'*40)
    return list(entities)

def count_hotwords_in_inference(row):
    manual_entities = row.manual_entities.split('|')
    inference = str(row.inference)
    
    hotwords_in_inference = 0
    for entity in manual_entities:
        if entity in inference:
            # print(entity)
            hotwords_in_inference += 1
    # print(hotwords_in_inference)
    return hotwords_in_inference

def count_hotwords_not_in_inference(row):
    manual_entities = row.manual_entities.split('|')
    inference = str(row.inference)
    
    hotwords_not_in_inference = 0
    for entity in manual_entities:
        if entity not in inference:
            # print(entity)
            hotwords_not_in_inference += 1
    # print(hotwords_not_in_inference)
    return hotwords_not_in_inference

def calculate_recall(row):
    return row.hotwords_in_inference / (row.hotwords_in_inference + row.hotwords_not_in_inference)


def unrecognized_entities(row):
    words = []
    for word in row.manual_entities.split('|'):
        if not word in str(row.inference):
            words.append(word)
    # print(words)
    return '|'.join(words)

def extract_hotwords_info(row):
    manual_transcription=row.Transcription

    manual_entities = extract_entities_from_reference(manual_transcription)
    return '|'.join(manual_entities)


def count_false_positives(row):
    hotwords_list = str(row.hotwords).split('|')
    inference  = str(row.inference)
    reference = str(row.Transcription).lower()

    false_positives = 0
    
    for word in hotwords_list:
        if word in inference and not word in reference:
            false_positives += 1

    # print(false_positives)

    return false_positives

def get_false_positives(row):
    hotwords_list = str(row.hotwords).split('|')
    inference  = str(row.inference)
    reference = str(row.Transcription).lower()

    false_positives = 0
    words = []
    for word in hotwords_list:
        if word in inference and not word in reference:
            words.append(word)

    # print(false_positives)

    return '|'.join(words)


def calculate_precision(row):
    fp = row.false_positive_hotwords
    tp = row.hotwords_in_inference
    try:
        return tp/(fp + tp)
    except ZeroDivisionError:
        return pd.NA

def calculate_wer(row):
    transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ]) 

    ground_truth = str(row.Transcription)
    hypothesis = str(row.inference)

    if len(ground_truth) <= 5 or len(hypothesis) <= 5:
        print('\tZERO')
        # return pd.NA
        return 0

    the_wer = jiwer.wer(
    ground_truth, 
    hypothesis, 
    truth_transform=transformation, 
    hypothesis_transform=transformation
    )
    # print(the_wer)
    return the_wer

def make_plot(precision:float, recall:float, F1:float, wers=None, folder = '.', macro=True):
    x = [i for i in range(len(precision))]

    fig, ax = plt.subplots()

    plt.plot(x, precision, label='precision')
    plt.plot(x, recall, label='recall')
    plt.plot(x, F1, label='F1')
    if wers:
        plt.plot(x, wers, label='WER')

    plt.ylim(0,1)
    if macro:
        plt.title("Hotwords Results")
    else:
        plt.title("Hotwords Results")

    plt.legend(loc='lower right')

    plt.vlines(x=3, ymin=0, ymax=1, color='black', linestyles='--', lw=0.5)

    plt.xlabel("Hotwords weight")

    outfile = f"{'macro' if macro else 'micro'}_average_results.png"
    outfile = os.path.join(folder, outfile)

    fig.savefig(outfile)
    print(f"Plot saved as {outfile}")

def make_data_filename(filename, folder):
    basename = os.path.basename(filename)
    barename, extension = os.path.splitext(basename)

    outname = f"{barename}_processed{extension}"

    outname = os.path.join(folder, outname)
    
    return outname

for root, folders,files in os.walk(root_folder):

    # print(files)
    files = sorted(files, key=lambda x:get_hotwords_weight(x))
    
    macroAR = []
    macroAP = []
    macro_f1s = []

    microAR = []
    microAP = []
    micro_f1s = []

    wers = []

    for file in files:
        if not file.endswith('.csv'):
            continue
        print(f"processing {file}")
        # input()
        hotwords_weight = get_hotwords_weight(file)

        data = pd.read_csv(os.path.join(root,file), sep='\t', index_col=0)

        # input(data.columns)
        data['manual_entities'] = data.apply(extract_hotwords_info, axis=1)
        # print(data)
        data['hotwords_in_inference']= data.apply(count_hotwords_in_inference, axis=1)
        data['hotwords_not_in_inference']= data.apply(count_hotwords_not_in_inference, axis=1)
        data['recall'] = data.apply(calculate_recall, axis=1)
        data['unrecognized_words'] = data.apply(unrecognized_entities, axis=1)
        data['false_positive_hotwords'] = data.apply(count_false_positives, axis = 1)
        data['false_positives_words'] = data.apply(get_false_positives, axis = 1)
        data['precision'] = data.apply(calculate_precision, axis = 1)
        data['WER'] = data.apply(calculate_wer, axis = 1)


        macro_average_recall = data.recall.mean()
        macro_stdev_recall = data.recall.std()
        macroAR.append(macro_average_recall)
        print(f'\tMacro AR: {macro_average_recall:0.2f} ({macro_stdev_recall:0.2f})')
        
        micro_average_recall = data.hotwords_in_inference.sum() / (data.hotwords_in_inference.sum() + data.hotwords_not_in_inference.sum())
        microAR.append(micro_average_recall)
        print(f'\tMicro AR: {micro_average_recall:0.2f}')
        print()

        macro_average_precision = data.precision.mean()
        macro_stdev_precision = data.precision.std()
        macroAP.append(macro_average_precision)
        print(f"\tMacro AP: {macro_average_precision:0.2f} ({macro_average_precision:0.2f})")
        micro_average_precision = data.hotwords_in_inference.sum() / (data.hotwords_in_inference.sum() + data.false_positive_hotwords.sum())
        microAP.append(micro_average_precision)
        print(f"\tMicro AP: {micro_average_precision:0.2f}")

        print()

        macro_F1 = 2* ( (macro_average_precision * macro_average_recall)/(macro_average_precision + macro_average_recall) )
        macro_f1s.append(macro_F1)
        print(f"\tMacro F1: {macro_F1:0.2f}")

        micro_F1 = 2* ( (micro_average_precision * micro_average_recall)/(micro_average_precision + micro_average_recall) )
        micro_f1s.append(micro_F1)
        print(f"\tMicro F1: {micro_F1:0.2f}")

        macro_average_wer = data.WER.mean()
        wers.append(macro_average_wer)
        print(f"\tMacro WER: {macro_average_wer:0.4f}")
        print()

        data_outfile = make_data_filename(file, TABLES_FOLDER)
        data.to_csv(data_outfile, sep='\t')
        print(f"new table saved as {data_outfile}")



    make_plot(macroAP, macroAR, macro_f1s, wers, PLOTS_FOLDER, macro=True)
    make_plot(microAP, microAR, micro_f1s, wers, PLOTS_FOLDER, macro=False)

        # print(hotwords)
