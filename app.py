import os
import transformers
import pandas as pd
import gradio


examples = [
    ['The English writer and the Afghani soldier.'],
    ['It was written by members of the United Nation.'],
    [('There were more than a hundred wolves in the Tiger Basin.  It is a dangerous place '
      'after 9 p.m., especially near Lake Victoria.')]]


# Pipeline
classifier = transformers.pipeline(
    task='ner', model=os.path.join(os.getcwd(), 'data', 'model'),
    config=os.path.join(os.getcwd(), 'data', 'model'),
    tokenizer=os.path.join(os.getcwd(), 'data', 'model'),
    device='cpu')


def custom(text):
    """

    :param text:
    :return:
    """

    tokens = classifier(text)
    summary = pd.DataFrame.from_records(data=tokens)
    summary = summary.copy()[['word', 'entity', 'score']]

    return {'text': text, 'entities': tokens}, summary.to_dict(orient='records'), tokens


with gradio.Blocks() as demo:

    gradio.Markdown(value=('<h1>Token Classification</h1><br><b>An illustration.</b>'), line_breaks=True)

    with gradio.Row():
        with gradio.Column(scale=3):
            text = gradio.Textbox(label='TEXT', placeholder="Enter sentence here...", max_length=2000)
        with gradio.Column(scale=2):
            detections = gradio.HighlightedText(label='DETECTIONS', interactive=False)
            scores = gradio.JSON(label='SCORES')
            compact = gradio.Textbox(label='COMPACT')
    with gradio.Row():
        detect = gradio.Button(value='Submit', variant='huggingface')
        gradio.ClearButton([text, detections, scores, compact], variant='huggingface')

    detect.click(custom, inputs=text, outputs=[detections, scores, compact])    
    gradio.Examples(examples=examples, inputs=[text], examples_per_page=1)

demo.launch(server_port=7860)