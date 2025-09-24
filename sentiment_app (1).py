
from transformers import pipeline
import gradio as gr

# Load sentiment pipeline
sentiment = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment(text)[0]
    label = result['label']
    score = round(result['score'], 4)
    return f"Sentiment: {label} (Confidence: {score})"

app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Type your text here..."),
    outputs="text",
    title="Sentiment Analysis App",
    description="Enter text and instantly get the sentiment (Positive/Negative)."
)

# app.launch()
app.launch(share=True) #for public link
