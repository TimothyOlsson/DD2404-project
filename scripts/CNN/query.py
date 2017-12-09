import numpy as np

def predict_to_AAseq(predictions, use_ascii):
    if use_ascii:
        predictions = [np.round(x*90)+65 for x in predictions]
        for i in range(len(predictions)):
            predictions_seq = predictions[i]
            predictions_seq = [chr(int(x)) for x in predictions_seq]
            yield predictions_seq
            
    elif not use_ascii:
        AA_dict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                   'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
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
        AA_dict = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                   'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
        AA_dict = {y: x for x, y in AA_dict.items()} # Reverse, so number is AA
        predictions = [np.round(x*19) for x in predictions]
        for i in range(len(predictions)):
            predictions_seq = predictions[i]
            predictions_seq = [str(int(x)) for x in predictions_seq]
            yield predictions_seq
