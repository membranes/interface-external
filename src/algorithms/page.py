"""Module page.py"""
import numpy as np
import pandas as pd


class Page:
    """
    Builds the page details of an input text.
    """

    def __init__(self, piece: str):
        """

        :param piece: A piece of text; composed of sentences or/and paragraphs
        """

        temporary = piece.splitlines()
        self.__text = ' '.join(temporary)

    def __splittings(self) -> list:
        """

        :return: list({'start': ,'word': }, ...)
        """

        points = enumerate(self.__text)
        text = ''
        place = np.empty(0, dtype=int)
        splittings = []

        for point in list(points):

            if point[1] == ' ' or point[1] == '(' or point[1] == ')':
                splittings.append({'start': place[0] if len(place) > 0 else np.nan, 'word': text})
                text = ''
                place = np.empty(0, dtype=int)
            else:
                place = np.append(place, point[0])
                text = ''.join([text, point[1]])

        splittings.append({'start': place[0] if len(place) > 0 else np.nan, 'word': text})

        return splittings

    @staticmethod
    def __page(splittings: list) -> pd.DataFrame:
        """

        :param splittings:
        :return: pandas.DataFrame(data={'start': ...,'word': ...,'end': ...})
        """

        frame = pd.DataFrame.from_records(data=splittings)
        frame = frame.copy().loc[frame['start'].notna(), :]

        # Setting the character indices as type integer
        frame['start'] = frame['start'].astype(dtype=int)
        frame['end'] = frame['start'] + frame['word'].str.len()

        # Ascertaining the words order
        frame.sort_values(by='start', inplace=True)

        return frame

    def exc(self):
        """

        :return:
        """

        splittings = self.__splittings()

        return self.__page(splittings=splittings)
