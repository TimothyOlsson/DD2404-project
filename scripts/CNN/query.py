import numpy as np

def predict_to_AAseq(predictions, use_ascii):
    if use_ascii:
        predictions = [np.round(x*90)+65 for x in predictions]
        for i in range(len(predictions)):
            predictions_seq = predictions[i]
            predictions_seq = [chr(int(x)) for x in predictions_seq]
            yield predictions_seq
            
    elif not use_ascii:
        AA_dict = {'R': 0, 'H': 1, 'K': 2, 'D': 3, 'E': 4, 'S': 5, 'T': 6, 'N': 7, 'Q': 8, 'C': 9,
                   'G': 10, 'P': 11, 'A': 12, 'V': 13, 'I': 14, 'L': 15, 'M': 16, 'F': 17, 'Y': 18, 'W': 19}
        AA_dict = {y: x for x, y in AA_dict.items()} # Reverse, so number is AA
        predictions = [np.round(x*19) for x in predictions]
        for i in range(len(predictions)):
            predictions_seq = predictions[i]
            predictions_seq = [AA_dict[int(x)] if 0. <= int(x) <= 19.
                                else 'X'
                                for x in predictions_seq]
            yield predictions_seq

def predict_to_numbers(predictions, use_ascii):
    if use_ascii:
        predictions = [np.round(x*90)+65 for x in predictions]
        for i in range(len(predictions)):
            predictions_seq = predictions[i]
            predictions_seq = [int(x) for x in predictions_seq]
            yield predictions_seq
            
    elif not use_ascii:
        AA_dict = {'R': 0, 'H': 1, 'K': 2, 'D': 3, 'E': 4, 'S': 5, 'T': 6, 'N': 7, 'Q': 8, 'C': 9,
                   'G': 10, 'P': 11, 'A': 12, 'V': 13, 'I': 14, 'L': 15, 'M': 16, 'F': 17, 'Y': 18, 'W': 19}
        AA_dict = {y: x for x, y in AA_dict.items()} # Reverse, so number is AA
        predictions = [np.round(x*19) for x in predictions]
        for i in range(len(predictions)):
            predictions_seq = predictions[i]
            predictions_seq = [str(int(x)) for x in predictions_seq]
            yield predictions_seq
