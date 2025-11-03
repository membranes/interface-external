"""Module app.py"""
import gradio
import pandas as pd
import transformers

import src.algorithms.interface
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


with gradio.Blocks() as demo:

    gradio.Markdown(value='<h1>Token Classification</h1><br><b>An illustration.</b>', line_breaks=True)

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

    detect.click(custom, inputs=text, outputs=[detections, scores, compact])    
    gradio.Examples(examples=configurations.examples, inputs=[text], examples_per_page=1)

demo.launch(server_port=7860)
