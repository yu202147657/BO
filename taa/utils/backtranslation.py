import torch

from transformers import MarianMTModel, MarianTokenizer

import pandas as pd
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# English to Romance languages
target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
target_tokenizer = MarianTokenizer.from_pretrained(target_model_name, seed=42)
target_model = MarianMTModel.from_pretrained(target_model_name).cuda()

# Romance languages to English
en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
en_tokenizer = MarianTokenizer.from_pretrained(en_model_name, seed=42)
en_model = MarianMTModel.from_pretrained(en_model_name).cuda()


def translate(texts, model, tokenizer, language="fr", num_beams=1):
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]
    encoded = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors='pt').to(device)
    translated = model.generate(**encoded, do_sample=True, max_length=512, top_k=0, num_beams=1, temperature=0.7)
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_texts


def back_translate(texts, source_lang="en", target_lang="fr", num_beams=1):
    fr_texts = translate(texts, target_model, target_tokenizer, language=target_lang, num_beams=num_beams)
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, language=source_lang, num_beams=num_beams)
    return back_translated_texts

def gen_back_translate(texts, batch_size):
    
    aug_texts = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        
        aug_text = back_translate(batch, source_lang="en", target_lang="fr",num_beams=1)
        aug_texts.extend(aug_text)
    
    return aug_texts