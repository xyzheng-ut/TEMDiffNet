import numpy as np
from pymatgen.core import Element
from typing import List, Union
import tensorflow as tf
MAX_ATOMIC_NUM = 100


def get_atomic_number(symbol: str) -> int:
    # print(symbol)
    """Get atomic number from Element symbol."""
    return Element(symbol).Z


class ChemicalSystemMultiHotEmbedding(tf.keras.Model):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Dense(hidden_dim)

    @staticmethod
    def _sequence_to_multi_hot(x: List[str]) -> tf.Tensor:
        """
        Converts a sequence of unique elements present in a single structure to a multi-hot
        vector of 1s (present) and 0s (not present) for each unique element.
        """
        # Convert element symbols to atomic numbers
        chemical_system_numbers = [get_atomic_number(symbol) for symbol in x]
        # Create a multi-hot vector
        chemical_system_condition = np.zeros((MAX_ATOMIC_NUM + 1,), dtype=np.float32)
        chemical_system_condition[chemical_system_numbers] = 1.0
        return tf.convert_to_tensor(chemical_system_condition.reshape(1, -1))

    @staticmethod
    def sequences_to_multi_hot(x: List[List[str]]) -> tf.Tensor:
        """
        Convert a list of sequences of unique elements present in a list of structures to a multi-hot tensor.
        """
        multi_hot_vectors = [ChemicalSystemMultiHotEmbedding._sequence_to_multi_hot(seq) for seq in x]
        return tf.concat(multi_hot_vectors, axis=0)

    @staticmethod
    def convert_to_list_of_str(x: list[str] | list[list[str]]) -> list[list[str]]:
        """
        Returns
        -------
        list[list[str]] -- a list of length n_structures_in_batch of chemical systems for each structure
            where the chemical system is specified as a list of unique elements in the structure.
        """
        if isinstance(x[0], str):
            # list[Sequence[str]]
            x = [_x.split("-") for _x in x if isinstance(_x, str)]
            # print(x)

        return x  # type: ignore

    def call(self, x: Union[List[str], List[List[str]]]) -> tf.Tensor:
        """
        Forward pass of the model.
        """
        x = self.convert_to_list_of_str(x)
        # print(x)
        multi_hot_representation = self.sequences_to_multi_hot(x)
        return self.embedding(multi_hot_representation)



# Get all defined elements
# from pymatgen.core.periodic_table import Element

# Get all valid chemical elements
# all_elements = [el.symbol for el in Element]
#
# print(all_elements[:101])
# print(len(all_elements[:101]))
#
# a =[['Li', 'He', 'Tm',"Al"],['Hg', 'Mg', 'Tm',"Al"]]
# # a = ["Cd-Mg-Sc","Cd-Mg-Sc"]
# #
# # # for c in a:
# # #     print(c)
# #
# b = ChemicalSystemMultiHotEmbedding.convert_to_list_of_str(a)
# print(b)
# ly = ChemicalSystemMultiHotEmbedding.sequences_to_multi_hot(b)
# print(ly)
