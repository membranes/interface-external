"""Module reconstruction.py"""
import pandas as pd


class Reconstruction:
    """
    Reconstruction
    """

    def __init__(self):
        """
        Constructor
        """

        self.__fields = ['word', 'tag_p', 'score', 'start', 'end']
        self.__rename = {'tag_p': 'entity'}

    def exc(self, mappings: pd.DataFrame) -> list[dict]:
        """

        :param mappings: ['start', 'word', 'end', 'code_of_tag_p', 'score', 'tag_p'] wherein '_p' denotes predicted.
        :return:
        """


        frame: pd.DataFrame = mappings.copy()[self.__fields]

        frame: pd.DataFrame = frame.copy().loc[frame['score'].notna(), :]
        frame.rename(columns=self.__rename, inplace=True)

        return frame.to_dict(orient='records')
