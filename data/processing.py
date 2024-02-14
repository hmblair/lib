# tokenizer.py

from typing import Iterable
import json
import os
from tqdm import tqdm

class CharacterTokenizer(dict):
    """
    A class for embedding sequences of characters into sequences of integers.
    """  
    def load(self, dict_dir : str) -> None:
        """
        Load the dictionary from a json file.

        Parameters
        ----------
        dict_dir : str
            The directory of the embedding dictionary.
        """
        if not dict_dir.endswith('.json'):
            raise ValueError('The dictionary file must be a json file.')
        if not os.path.exists(dict_dir):
            raise ValueError(f'The dictionary file {dict_dir} does not exist.')
        with open(dict_dir, 'r') as f:
            self.update(json.load(f))
    

    def __call__(
            self, 
            seqs : Iterable[str], 
            prog_bar : bool = False,
            ) -> list[list[int]]:
        """
        Embed a sequence of strings into a sequence of sequences of integers.

        Parameters
        ----------
        seqs : Iterable[str]
            An iterable of strings.

        Returns
        -------
        list[list[int]]
            A list of lists of integers, with each sublist corresponding to the
            embeddings of a single sequence.
        """
        iterable = tqdm(seqs) if prog_bar else seqs
        return [[self[s] for s in seq] for seq in iterable]