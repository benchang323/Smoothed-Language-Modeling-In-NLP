#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
from integerize import Integerizer   # look at integerize.py for more info

# Needed for Python's optional type annotations.
# We've included type annotations and recommend that you do the same, 
# so that mypy (or a similar package) can catch type errors in your code.
from typing import List, Optional

try:
    # PyTorch is your friend. Not *using* it will make your program so slow.
    # And it's also required for this assignment. ;-)
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    import torch as th
    import torch.nn as nn
except ImportError:
    print("\nERROR! You need to install Miniconda, then create and activate the nlp-class environment.  See the INSTRUCTIONS file.\n")
    raise


log = logging.getLogger(Path(__file__).stem)  # The only okay global variable.

# Logging is in general a good practice to check the behavior of your code
# while it's running. Compared to calling `print`, it provides two benefits.
# 
# - It prints to standard error (stderr), not standard output (stdout) by
#   default.  So these messages will normally go to your screen, even if
#   you have redirected stdout to a file.  And they will not be seen by
#   the autograder, so the autograder won't be confused by them.
# 
# - You can configure how much logging information is provided, by
#   controlling the logging 'level'. You have a few options, like
#   'debug', 'info', 'warning', and 'error'. By setting a global flag,
#   you can ensure that the information you want - and only that info -
#   is printed. As an example:
#        >>> try:
#        ...     rare_word = "prestidigitation"
#        ...     vocab.get_counts(rare_word)
#        ... except KeyError:
#        ...     log.error(f"Word that broke the program: {rare_word}")
#        ...     log.error(f"Current contents of vocab: {vocab.data}")
#        ...     raise  # Crash the program; can't recover.
#        >>> log.info(f"Size of vocabulary is {len(vocab)}")
#        >>> if len(vocab) == 0:
#        ...     log.warning(f"Empty vocab. This may cause problems.")
#        >>> log.debug(f"The values are {vocab}")
#   If we set the log level to be 'INFO', only the log.info, log.warning,
#   and log.error statements will be printed. You can calibrate exactly how 
#   much info you need, and when. None of these pollute stdout with things 
#   that aren't the real 'output' of your program.
# 
# In `parse_args`, we provided two command line options to control the logging level.
# The default level is 'INFO'. You can lower it to 'DEBUG' if you pass '--verbose'
# and you can raise it to 'WARNING' if you pass '--quiet'.
#
# More info: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
# 
# In all the starter code for the NLP course, we've elected to create a separate
# logger for each source code file, stored in a variable named log that
# is globally visible throughout the file.  That way, calls like log.info(...)
# will use the logger for the current source code file and thus their output will 
# helpfully show the filename.  You could configure the current file's logger using
# log.basicConfig(...), whereas logging.basicConfig(...) affects all of the loggers.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

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

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words(bagpipe)
    """

    def __init__(self) -> None:
        """Load information into coupled word-index mapping and embedding matrix."""
        # FINISH THIS FUNCTION

        # Store your stuff! Both the word-index mapping and the embedding matrix.
        #
        # Do something with this size info?
        # PyTorch's th.Tensor objects rely on fixed-size arrays in memory.
        # One of the worst things you can do for efficiency is
        # append row-by-row, like you would with a Python list.
        #
        # Probably make the entire list all at once, then convert to a th.Tensor.
        # Otherwise, make the th.Tensor and overwrite its contents row-by-row.
        self.row_num=0
        self.col_num=0
        self.integerizer=None
        self.lex=None
        self.cos_sim=None
    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        # FINISH THIS FUNCTION

        with open(file) as f:
            first_line = next(f)  # Peel off the special first line.
            first_line_ls=first_line.split()
            ls_dynamic=[]
            word_ls=[]
            for line in f:  # All of the other lines are regular.
                line_ls=line.split()  # `pass` is a placeholder. Replace with real code!
                word_ls.append(line_ls[0])
                ls_dynamic.append([float(x) for x in line_ls[1:]])
                 
            

        lexicon = Lexicon()  # Maybe put args here. Maybe follow Builder pattern.
        lexicon.integerizer=Integerizer(word_ls)
        lexicon.row_num=int(first_line_ls[0])
        lexicon.col_num=int(first_line_ls[1])
        lexicon.lex = th.tensor(ls_dynamic)


        

        return lexicon

    def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""
        # FINISH THIS FUNCTION

        # The star above forces you to use `plus` and `minus` only
        # as named arguments. This helps avoid mixups or readability
        # problems where you forget which comes first.
        #
        # We've also given `plus` and `minus` the type annotation
        # Optional[str]. This means that the argument may be None, or
        # it may be a string. If you don't provide these, it'll automatically
        # use the default value we provided: None.
        if (minus is None) != (plus is None):  # != is the XOR operation!
            raise TypeError("Must include both of `plus` and `minus` or neither.")
        # Keep going!
        # Be sure that you use fast, batched computations
        # instead of looping over the rows. If you use a loop or a comprehension
        # in this function, you've probably made a mistake.

        input_i=self.integerizer.index(word)
        this_lex=self.lex
        if minus!=None and plus!=None:
            plus_i=self.integerizer.index(plus)
            minus_i=self.integerizer.index(minus)
            word_v=self.lex[input_i]
            plus_v=self.lex[plus_i]
            minus_v=self.lex[minus_i]
            new_word_v=th.add(th.sub(word_v, minus_v),plus_v)
            this_lex[input_i]=new_word_v



        
        
        eps=1e-8
        lex_norm=this_lex.norm(dim=1)[:,None]
        #print(lex_norm)
        #print(eps * th.ones_like(lex_norm))
        #print('max',th.max(lex_norm, eps * th.ones_like(lex_norm)))
        lex_nmed=this_lex / th.max(lex_norm, eps * th.ones_like(lex_norm))
        target_v=lex_nmed[input_i]
        #print('input_i',input_i)
        #print('shape',lex_nmed.size(), target_v.size())
        self.cos_sim=th.mv(lex_nmed,target_v)



        NUMOFSIMILARWORDS=1
        similar_words=[]
        
        
        cos=self.cos_sim.tolist()

        cos_th=th.nn.CosineSimilarity(dim=0, eps=1e-08)
        # new_cos=[]
        # for i in range(self.row_num):
        #     new_cos.append(cos_th(self.lex[input_i],self.lex[i]))
        # #print(cos_th(self.lex[input_i],self.lex[0]))
        # print(new_cos[:10])
        # print(cos[:10])
        # print(cos[1000:1050])

        # por_v=self.lex[self.integerizer.index('portland')]
        # sea_v=self.lex[self.integerizer.index('seattle')]
        # print(por_v)
        # print(sea_v)
        #print(cos)
        
        # print('i of portland',self.integerizer.index('portland'))
        # print(self.integerizer[9335],cos[9335],self.cos_sim[9335])
        # print(self.integerizer[36122],cos[36122],self.cos_sim[36122])
        # print(self.integerizer[4964],cos[4964],self.cos_sim[4964])
        
        # print('sorted')
        # print(sorted(cos)[-20:])
        # print(self.cos_sim.tolist()[:10])
        # print(cos[:10])
        biggest=set()
        biggest.add(input_i)
        for i in range(NUMOFSIMILARWORDS):
            max=-5
            max_i=0
            for j in range(len(cos)):

                this_cos=cos[j]
                # if this_cos < -0.6:
                #     print(this_cos)
                if this_cos >= max and not(j in biggest):
                    max=this_cos
                    max_i=j
                    
            biggest.add(max_i)
            
            #print(max,len(cos))
            # print(similar_words)
            similar_words.append(self.integerizer[max_i])
        

        


        #[[-10.0,32.0,-4.0,2.0],[2.0,4.0,6.0,8.0],[3.0,5.0,7.0,9.0]]
        #input1 = th.tensor( [[1.0,2.0,3.0,4.0],[2.0,4.0,6.0,8.0],[3.0,5.0,7.0,9.0]])
        #input2 = th.tensor([-10.0,32.0,-4.0,2.0])
        # i=0
        
        # input1_norm=input1.norm(dim=1)[:, None]
        # input1_nmed = input1 / th.max(input1_norm, eps * th.ones_like(input1_norm))
        # cos_sim = th.mm(input1_nmed, th.t(input1_nmed))
        # print(cos_sim)
        # DotProducts_M=th.mm(input1,th.t(input1))
    
        # print(DotProducts_M)

        #torch.norm(input, p='fro', dim=None,
        # Norms=th.norm(input1, p=2,dim=1)  
    
        # print(type(cos))
        # output = cos(input1, input2)
        # print('sample cos similartity',output)











        return similar_words


def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(" ".join(similar_words))  # print all words on one line, separated by spaces


if __name__ == "__main__":
    main()
