from flask import Flask, render_template, jsonify, request
import openai
from pygame import mixer
from happytransformer import  HappyTextToText

import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

#Setup for Paraphrasing of the input
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
app = Flask(__name__)

# Set up your OpenAI API key
openai.api_key = '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'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process():
    data = request.get_json()
    text = data['message']
    option = data['option']

    if option == 'grammar':
        corrected_text = correct_grammar(text)
        return jsonify({'result': corrected_text})
    elif option == 'paraphrase':
        result = paraphrase(text, num_return_sequences=4)
        output_string = '\n\n'.join(result)
        return jsonify({'result': output_string})
    
    return jsonify({'result': ''})

def correct_grammar(text):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=f"Correct the grammar of the following text and do not return anything except the corrected grammar:\n{text}",
        max_tokens=100,
        temperature=0.8,
        n=1,
        stop=None,
        logprobs=0,
        echo=False
    )
    corrected_text = response.choices[0].text.strip()
    return corrected_text
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

def paraphrase(text,num_return_sequences=4):
  batch = tokenizer.prepare_seq2seq_batch([text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=10, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

if __name__ == '__main__':
    app.run(debug=True)
