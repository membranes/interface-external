"""Module detections.py"""
import numpy as np
import pandas as pd


class Detections:
    """
    Detections
    """

    def __init__(self, tokens: list):
        """

        :param tokens: The words/word-pieces that have been classified, i.e., the model
                       estimates that they are not miscellaneous entities.
                       list({'entity': ..., 'score': ..., 'index': ..., 'word': ..., 'start': ..., 'end': ...},
                            ...)
        """

        self.__tokens = tokens

    def __get_data(self) -> pd.DataFrame:
        """

        :return:
        """

        data = pd.DataFrame.from_records(self.__tokens)
        data.sort_values(by='index', inplace=True)

        return data

    @staticmethod
    def __anomaly(blob: pd.DataFrame) -> pd.DataFrame:
        """
        If a word is split into word-pieces their character indices overlap at the edge.  This
        function ensures mutually exclusive indices.

        :param blob:
        :return:
        """

        data = blob.copy()
        data =  data.assign(begin = data['start'] + (data['word'].str.startswith('##')).astype(int) )
        data['indices'] = data.apply(
            lambda x: np.linspace(x['begin'], x['end'], x['end'] - x['begin'] + 1, dtype=int),
            axis=1)

        return data

    def exc(self, m_config: dict):
        """

        :param m_config: The underlying model's configuration dictionary.
        :return:
        """

        data = self.__get_data()
        data = self.__anomaly(blob=data)

        data['code_of_tag_p'] = data['entity'].map(m_config['label2id'])

        return data
