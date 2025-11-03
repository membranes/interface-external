"""Module mappings.py"""
import typing

import numpy as np
import pandas as pd


class Mappings:
    """
    Of input text page and detections.
    """

    def __init__(self, page: pd.DataFrame, detections: pd.DataFrame):
        """

        :param page: The input text page in data frame form.  The text has been split into words; each
                     instance represents a word.
        :param detections: The non-miscellaneous tokens detected by the model.
        """

        self.__page = page
        self.__detections = detections

    def __intersects(self, instance: np.ndarray) -> pd.Series:
        """

        :param instance: Part of an instance of self.__page.  It consists of an instance's
                         word start index [0], and word end index [1]
        :return:
        """

        # Do any of the tokens of self.__detections classify this instance?
        indices_of_instance = np.linspace(instance[0], instance[1], instance[1] - instance[0] + 1, dtype=int)
        conditionals = self.__detections['indices'].apply(lambda x: np.isin(x, indices_of_instance).any())

        return conditionals

    def __instances(self, conditionals) -> typing.Tuple[pd.DataFrame, int]:
        """

        :param conditionals:
        :return:
        """

        instances = self.__detections.copy().loc[conditionals, :]
        instances.sort_values(by='index', ascending=True, inplace=True)

        # The distinct categories
        n_entities = instances['entity'].unique().shape[0]

        return instances, n_entities

    def __code_of_tag(self, instance: np.ndarray) -> int:
        """

        :param instance:
        :return:
        """

        conditionals = self.__intersects(instance=instance)

        if sum(conditionals) == 0:
            return -1

        instances, n_entities = self.__instances(conditionals=conditionals)

        if n_entities == 1:
            return instances['code_of_tag_p'].to_numpy()[0]

        return -1

    def __score(self, instance: np.ndarray) -> float:
        """

        :param instance:
        :return:
        """

        conditionals = self.__intersects(instance=instance)

        if sum(conditionals) == 0:
            return np.nan

        instances, n_entities = self.__instances(conditionals=conditionals)

        if n_entities == 1:
            return instances['score'].to_numpy().prod()

        return np.nan

    def exc(self, m_config: dict):
        """

        :param m_config: The underlying model's configuration dictionary.
        :return:
        """

        data = self.__page.copy()

        # Mapping to detected tokens
        data['code_of_tag_p'] = np.apply_along_axis(func1d=self.__code_of_tag, axis=1, arr=data[['start', 'end']])
        data['code_of_tag_p'] = data['code_of_tag_p'].where(data['code_of_tag_p'] > -1, m_config['label2id']['O'])

        # Mapping to scores thereof
        data['score'] = np.apply_along_axis(func1d=self.__score, axis=1, arr=data[['start', 'end']])

        # Finally.  An NaN value should not exist; just in case.
        data['tag_p'] = data['code_of_tag_p'].map(lambda x: m_config['id2label'][str(int(x))], na_action='ignore')

        return data
