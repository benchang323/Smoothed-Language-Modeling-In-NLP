#!/usr/bin/env python3

import argparse
import logging
import math
from pathlib import Path
from probs import Wordtype, LanguageModel, num_tokens, read_trigrams  # Assuming probs module contains necessary functionalities

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speech Recognition Program")
    parser.add_argument("model", type=Path, help="path to the trained model")
    parser.add_argument("test_files", type=Path, nargs="*")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_const", const=logging.DEBUG, default=logging.INFO)
    verbosity.add_argument("-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING)
    return parser.parse_args()

def read_transcriptions(file: Path):
    with file.open() as f:
        num_transcriptions = int(f.readline().split()[0])  # Get the number of transcriptions
        transcriptions = []
        for _ in range(num_transcriptions):
            line = f.readline().split()
            wer, num_words = float(line[0]), int(line[1])  # Word error rate and number of words
            transcription = " ".join(line[2:])  # The transcription
            transcriptions.append((wer, num_words, transcription))
        return transcriptions

def choose_best_transcription(transcriptions, lm: LanguageModel):
    best_log_prob = float('-inf')
    best_wer = 1.0
    for wer, _, transcription in transcriptions:
        log_prob = 0.0
        tokens = transcription.split()
        x: Wordtype; y: Wordtype; z: Wordtype
        for i in range(2, len(tokens)):
            x, y, z = tokens[i-2], tokens[i-1], tokens[i]
            log_prob += lm.log_prob(x, y, z)
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_wer = wer
    return best_wer

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lm = LanguageModel.load(args.model)
    
    total_wer = 0.0
    total_words = 0
    for file in args.test_files:
        transcriptions = read_transcriptions(file)
        best_wer = choose_best_transcription(transcriptions, lm)
        print(f"{best_wer:.3f}\t{file.stem}")
        total_wer += best_wer * transcriptions[0][1]  # Use the num_words of the first transcription for overall WER calculation
        total_words += transcriptions[0][1]
    
    overall_wer = total_wer / total_words
    print(f"{overall_wer:.3f}\tOVERALL")

if __name__ == "__main__":
    main()
