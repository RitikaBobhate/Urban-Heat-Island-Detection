import gradio as gr
from transformers import pipeline

pipe = pipeline("text2text-generation", model="google/flan-t5-small")

def chat(prompt):
    out = pipe(prompt, max_new_tokens=100)
    return out[0]['generated_text']

demo = gr.Interface(fn=chat, inputs="text", outputs="text")
demo.launch()