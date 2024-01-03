#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
   

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "sample_num",
        type=int,
        help="number of samples user desires",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of sample sentences"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0


    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)

        # If the factor p(z | xy) = 0, then it will drive our cumulative file 
        # probability to 0 and our cumulative log_prob to -infinity.  In 
        # this case we can stop early, since the file probability will stay 
        # at 0 regardless of the remaining tokens.
        if log_prob == -math.inf: break 

        # Why did we bother stopping early?  It could occasionally
        # give a tiny speedup, but there is a more subtle reason -- it
        # avoids a ZeroDivisionError exception in the unsmoothed case.
        # If xyz has never been seen, then perhaps yz hasn't either,
        # in which case p(next token | yz) will be 0/0 if unsmoothed.
        # We can avoid having Python attempt 0/0 by stopping early.
        # (Conceptually, 0/0 is an indeterminate quantity that could
        # have any value, and clearly its value doesn't matter here
        # since we'd just be multiplying it by 0.)

    return log_prob


def main():
    # ./trigram_randsent.py [mymodel] 10 --max_length 20
    # to get 10 samples of length at most 20.
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    log.info("Testing...")
    lm = LanguageModel.load(args.model)

    # print(args.model)
    # print(args.sample_num)
    # print(args.max_length)


    #  ('<END_QUOTE>', 'EOS')     ('&NAME', 'EOS')

    # print(list(lm.candidates.items())[:10])
    # print('--------------------------------------------')
    # print(list(lm.context_count.keys())[:10])
    # print(list(lm.event_count.keys())[:10])



    # ['BOS', 'BOS', 'SUBJECT:', 'OOV', 'OOV', 'OOV', 'and', 'when', 'you', 'are', '&NAME', '&NAME', '.', 'I', 
    # 'intend', 'to', 'come', 'to', '.', 'I', 'need', 'a', 'healthy', 'supply', 'of', 'vitamins', 'and',
    #  'a', 'OOV', '&NAME', 'EOS']
    # print('test:')
    # print(lm.context_count[('&NAME', 'EOS') ])
    # print(lm.context_count[('OOV', '&NAME') ])
    # print(lm.context_count[('OOV', '&NAME', 'EOS') ])
    #print(lm.candidates[('&NAME', 'EOS')])

    for i in range(args.sample_num):

        sentence=[]
        context=('BOS','BOS')
        sentence=sentence+list(context)
        while not(sentence[-1]=='EOS'):
            context=tuple(sentence[-2:])
            # if context==('EOS','EOS'):
            #     print('changes')
            #     context=('BOS','BOS')
            sentence.append(lm.sample(context))
        sentence=sentence[2:]
        if len(sentence) > args.max_length:
            sentence=sentence[:args.max_length]
            sentence.append('...')
        sent=""
        for s in sentence:
            sent=sent+' '+s
    
        print(sent[1:])
    
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # We'll print that first.

    # log.info("Per-file log-probabilities:")
    # total_log_prob = 0.0
    # for file in args.test_files:
    #     log_prob: float = file_log_prob(file, lm)
    #     print(f"{log_prob:g}\t{file}")
    #     total_log_prob += log_prob


    # But cross-entropy is conventionally measured in bits: so when it's
    # time to print cross-entropy, we convert log base e to log base 2, 
    # by dividing by log(2).

    # bits = -total_log_prob / math.log(2)   # convert to bits of surprisal
    # tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    # print(f"Overall cross-entropy:\t{bits / tokens:.5f} bits per token")


if __name__ == "__main__":
    main()
