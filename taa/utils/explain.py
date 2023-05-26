import lime
from lime.lime_text import LimeTextExplainer

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig

from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

import numpy as np
import os


def explain(text):
    # Define predict function for your BERT model
    def predict(text):
        input_ids = torch.tensor([tokenizer.encode(text)])
        logits = model(input_ids)[0]
        return logits.detach().numpy()

    # Apply LIME explainer to text
    exp = explainer.explain_instance(text, predict, num_features=10, num_samples=5000)
    # Get LIME weights
    weights = np.array([x[1] for x in exp.as_list()])
    # Normalize weights
    weights /= np.sum(np.abs(weights))
    return weights



def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)    



class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)



def interpret_sentence(model_wrapper, model, tokenizer, vis_data_records_ig, sentence, label=1):

    model_wrapper.eval()
    model_wrapper.zero_grad()
    
    input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
    input_embedding = model_wrapper.model.bert.embeddings(input_ids.to('cuda'))
    
    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    ig = IntegratedGradients(model_wrapper)
    attributions_ig, delta = ig.attribute(input_embedding, n_steps=10, return_convergence_delta=True)

    print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy().tolist())    
    
    vis_data_records_ig = add_attributions_to_visualizer(attributions_ig.cpu(), tokens, pred, pred_ind, label, delta, vis_data_records_ig)
    
    return vis_data_records_ig
    
    
    
def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()
    
    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            "label",
                            attributions.sum(),       
                            tokens[:len(attributions)],
                            delta))    

    return vis_data_records

def integrated_gradient(texts, labels, model, abspath, file_name):
    """
    Take in file_name and df
    Return html with sentence attributions
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    bert_model_wrapper = BertModelWrapper(model)

    # accumalate couple samples in this array for visualization purposes
    vis_data_records_ig = []
    
    for i, text in enumerate(texts):
        vis_data_records_ig = interpret_sentence(bert_model_wrapper, model, tokenizer, vis_data_records_ig, sentence=text, label=labels[i])
        htm = visualization.visualize_text(vis_data_records_ig)

        with open(os.path.join(abspath, '%s.html' %
                                     (file_name)), "w") as file:
            file.write(htm.data)
            
def lime_explainer(texts, labels, model, class_names, abspath, file_name):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    explainer = LimeTextExplainer(class_names=class_names)
    
    def predictor(texts):
        outputs = model(**tokenizer(texts, return_tensors="pt", padding=True).to('cuda'))
        tensor_logits = outputs[0]
        probas = F.softmax(tensor_logits).detach().cpu().numpy()
        return probas
    
    for i, text in enumerate(texts):
        tokenizer(text, return_tensors='pt', padding=True)
        exp = explainer.explain_instance(text, predictor, num_features=20, num_samples=20)
        #exp.save_to_file(f"{abspath}/results/explain/{file_name}_{i}.html")
        exp.save_to_file(os.path.join(abspath, '%s_%d.html' %
                                     (file_name, i)))
        
    
    
if __name__ == '__main__':
        
    # Load BERT model and tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    texts = ['You are a poo face', 'How are you doing', 'Text o classify']
    labels = [1, 0, 0]
    
    integrated_gradient(texts, labels, model, 'test')
    
    class_names = ['pos', 'neg']
    lime_explainer(texts, labels, model, class_names, 'test')

    