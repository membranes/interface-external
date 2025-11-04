"""Module interface.py"""
import logging
import string

import pandas as pd

import src.algorithms.detections
import src.algorithms.mappings
import src.algorithms.page
import src.algorithms.reconstruction
import src.config
import src.functions.objects


class Interface:
    """
    Executes the functions that process the input text, and the tokens classifications results.
    """

    def __init__(self):
        """
        Constructor
        """

        self.__configurations = src.config.Config()

        # Characters space
        self.__characters = string.ascii_lowercase + string.digits + string.ascii_uppercase

    def __m_config(self) -> dict:
        """

        :return:
        """

        objects = src.functions.objects.Objects()

        return objects.read(uri=self.__configurations.config_)

    @staticmethod
    def __get_mappings(page: pd.DataFrame, tokens:list, m_config: dict) -> pd.DataFrame:
        """

        :param page:
        :param tokens:
        :param m_config:
        :return:
        """

        # The detections
        detections = src.algorithms.detections.Detections(tokens=tokens).exc(m_config=m_config)

        # Hence, map
        mappings = src.algorithms.mappings.Mappings(page=page, detections=detections).exc(m_config=m_config)

        return mappings

    def exc(self, piece: str, tokens: list) -> list:
        """

        :param piece:
        :param tokens:
        :return:
        """

        # The underlying model's configuration dictionary.
        m_config = self.__m_config()

        # The input piece
        page = src.algorithms.page.Page(piece=piece).exc()

        # If tokens is empty ...
        if len(tokens) == 0:
            return []

        # Else
        mappings = self.__get_mappings(page=page, tokens=tokens, m_config=m_config)

        return src.algorithms.reconstruction.Reconstruction().exc(mappings=mappings)
