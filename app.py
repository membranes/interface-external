"""Module app.py"""
import logging
import gradio
import pandas as pd
import transformers
import subprocess

import src.algorithms.interface
import src.functions.cache
import src.config

# Pipeline
configurations = src.config.Config()

# noinspection PyTypeChecker
classifier = transformers.pipeline(task='ner', model=configurations.model_, device='cpu')


def custom(piece):
    """

    :param piece:
    :return:
    """

    tokens = classifier(piece)

    # Reconstructing & Persisting
    tokens = tokens if len(tokens) == 0 else src.algorithms.interface.Interface().exc(
        piece=piece, tokens=tokens)

    # Summary
    summary = pd.DataFrame.from_records(data=tokens)
    summary = summary.copy()[['word', 'entity', 'score']] if not summary.empty else summary

    return {'text': piece, 'entities': tokens}, summary.to_dict(orient='records'), tokens

def __kill() -> str:
    """

    :return:
    """

    src.functions.cache.Cache().exc()

    return subprocess.check_output('kill -9 $(lsof -t -i:7860)', shell=True, text=True)

with gradio.Blocks() as demo:

    gradio.Markdown(value=('<h1>Token Classification</h1><br><b>An illustrative interactive interface; the '
                           'interface software allows for advanced interfaces.</b><br><br>The classes are '
                           '<b>organisation</b> (b-org, i-org), <b>person</b> (b-per, i-per), '
                           '<b>time</b> (b-tim, i-tim), <b>geographic entity</b> (b-geo, i-geo), <br>'
                           '<b>geo-political entity</b> (b-gpe, i-gpe).  The letter <b>b</b> '
                           'denotes <i><b>beginning</b></i>, whilst <b>i</b> denotes <i><b>inside</b></i>.<br>'), line_breaks=True)

    with gradio.Row():
        with gradio.Column(scale=3):
            text = gradio.Textbox(label='TEXT', placeholder="Enter sentence here...", lines=23, max_length=4000)
        with gradio.Column(scale=2):
            detections = gradio.HighlightedText(label='DETECTIONS', interactive=False)
            scores = gradio.JSON(label='SCORES')
            compact = gradio.Textbox(label='COMPACT')
    with gradio.Row():
        detect = gradio.Button(value='Submit', variant='huggingface')
        gradio.ClearButton([text, detections, scores, compact], variant='secondary')
        stop = gradio.Button('Disconnect', variant='stop', visible=True, size='lg')

    detect.click(custom, inputs=text, outputs=[detections, scores, compact])
    stop.click(fn=__kill)
    gradio.Examples(examples=configurations.examples, inputs=[text], examples_per_page=1)

demo.launch(server_port=7860)
