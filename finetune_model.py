import torch
import pandas as pd
import math
import random

from indicnlp import common
common.set_resources_path("./indic_nlp_resources")

from indicnlp import loader
loader.load()
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

from transformers import AlbertTokenizer, MBartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import sys
import numpy as np

from transformers import BertTokenizer,BertForMaskedLM


tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART")

bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")
# To get lang_id use any of ['<2as>', '<2bn>', '<2en>', '<2gu>', '<2hi>', '<2kn>', '<2ml>', '<2mr>', '<2or>', '<2pa>', '<2ta>', '<2te>']

# First tokenize the input and outputs. 
# The format below is how IndicBART was trained so the input should be "Sentence </s> <2xx>" where xx is the language code. 
# Similarly, the output should be "<2yy> Sentence </s>". 

input_sentence = "I am a boy"
output_sentence = "मैं एक लड़का हूँ"

# The input and output sentence is what the model needs for learning.
# Current example is translation but consider the following examples:
# Summarization: 
# Input is "There was a child. She liked food. She decided to learn to make food. She became a world class chef. She won an award." 
# Output is "A child learned cooking and won an award."
# Paraphrasing: 
# Input is "I love to eat food."
# Output is "I am a foodie"

inp = tokenizer(input_sentence, add_special_tokens=False, return_tensors="pt", padding=True).input_ids

out = tokenizer(output_sentence, add_special_tokens=False, return_tensors="pt", padding=True).input_ids

print("Original input sentence:", input_sentence)
print("Segmented input sentence:", tokenizer.convert_ids_to_tokens(inp[0]))
print("Input sentence as tensor: ", inp)
print("Original output sentence:", output_sentence)
print("Segmented output sentence:", tokenizer.convert_ids_to_tokens(out[0]))
print("Output sentence as tensor:", out)

# Load pre-trained model (weights)
with torch.no_grad():
    model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART")
    #model = BertForMaskedLM.from_pretrained('bert-large-cased')
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)
    #tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
def score(sentence):
    tensor_input = tokenizer(sentence, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
    #tokenize_input = ["[CLS]"]+tokenize_input+["[SEP]"]
    #tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        loss=model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())
 
## Batching

def yield_corpus_indefinitely_bi(corpus, language):
    """This shuffles the corpus at the beginning of each epoch and returns sentences indefinitely."""
    epoch_counter = 0
    num_lines = len(corpus)
    num_sentences_before_sort = 5000
    num_sorted_segments = (num_lines // num_sentences_before_sort) + 1
    while True:
        print("Shuffling corpus:", language)
        random.shuffle(corpus)
        for src_line, tgt_line in corpus:
            yield src_line, tgt_line
        epoch_counter += 1
        print("Finished epoch", epoch_counter, "for language:", language)
    return None, None ## We should never reach this point.


def generate_batches_bilingual(tok, num_batches, is_summarization=False, batch_size=16, src_lang="en", tgt_lang="hi", src_file_prefix="TED2020.en-hi", tgt_file_prefix="TED2020.en-hi"):
    """Generates the source, target and source attention masks for the training set. 
    The source and target sentences are ignored if empty and are truncated if longer than a 
    threshold. The batch size in this context is the maximum number of tokens in the batch post padding."""
    batch_count = 0
    mask_tok = "[MASK]"

    language_list = [src_lang+"-"+tgt_lang]
    print("Training for:", language_list)
    src_file_content = open(src_file_prefix).readlines()
    tgt_file_content = open(tgt_file_prefix).readlines()
    file_content = list(zip(src_file_content, tgt_file_content))
    file_iterator = yield_corpus_indefinitely_bi(file_content, src_lang+"-"+tgt_lang)
    slang = "<2"+src_lang+">"
    tlang = "<2"+tgt_lang+">"
            
    while batch_count != num_batches:
        curr_batch_count = 0
        encoder_input_batch = []
        decoder_input_batch = []
        decoder_label_batch = []
        batch_count += 1
        max_src_sent_len = 0
        max_tgt_sent_len = 0
        sents_in_batch = 0
        while True:
            src_sent, tgt_sent = next(file_iterator)
            src_sent = src_sent.strip()
            tgt_sent = tgt_sent.strip()
            # if slang != "<2en>" and slang != "<2hi>": # Transliterate to Devanagari
            #     src_sent = UnicodeIndicTransliterator.transliterate(src_sent, slang[2:4], "hi")
            # if tlang != "<2en>" and tlang != "<2hi>": # Transliterate to Devanagari
            #     tgt_sent = UnicodeIndicTransliterator.transliterate(tgt_sent, tlang[2:4], "hi")
            src_sent_split = src_sent.split(" ")
            tgt_sent_split = tgt_sent.split(" ")
            tgt_sent_len = len(tgt_sent_split)
            src_sent_len = len(src_sent_split)
            
            if src_sent_len < 1 or tgt_sent_len < 1:
                continue
            else:   # Initial truncation
                if src_sent_len >= 100:
                    src_sent_split = src_sent_split[:100]
                    src_sent = " ".join(src_sent_split)
                    src_sent_len = 100
                if tgt_sent_len >= 100:
                    tgt_sent_split = tgt_sent_split[:100]
                    tgt_sent = " ".join(tgt_sent_split)
                    tgt_sent_len = 100
            
            iids = tok(src_sent + " </s> " + slang, add_special_tokens=False, return_tensors="pt").input_ids
            curr_src_sent_len = len(iids[0])
            if curr_src_sent_len > 256:
                src_sent = tok.decode(iids[0][0:256-2], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                curr_src_sent_len = 256
            
            iids = tok(tlang + " " + tgt_sent, add_special_tokens=False, return_tensors="pt").input_ids
            curr_tgt_sent_len = len(iids[0])
            if curr_tgt_sent_len > 256:
                tgt_sent = tok.decode(iids[0][1:256], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                curr_tgt_sent_len = 256
            
            
            encoder_input_batch.append(src_sent + " </s> " + slang)
            decoder_input_batch.append(tlang + " " + tgt_sent)
            decoder_label_batch.append(tgt_sent + " </s>")
            sents_in_batch += 1
            if sents_in_batch == batch_size:
                    break
                
        input_ids = tok(encoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        input_masks = (input_ids != 4).int()
        decoder_input_ids = tok(decoder_input_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        labels = tok(decoder_label_batch, add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        yield input_ids, input_masks, decoder_input_ids, labels

## Loss

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None):
    """From fairseq. This returns the label smoothed cross entropy loss."""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
        denominator = (1.0 - 1.0*pad_mask)
        denominator = denominator.sum()
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
        denominator = 1.0
    
    if ignore_index is not None:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    else:
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()
        
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    loss = loss/denominator
    return loss

tokenizer = AlbertTokenizer.from_pretrained("ai4bharat/IndicBART", do_lower_case=False, use_fast=False, keep_accents=True)

bos_id = tokenizer._convert_token_to_id_with_added_voc("<s>")
eos_id = tokenizer._convert_token_to_id_with_added_voc("</s>")
pad_id = tokenizer._convert_token_to_id_with_added_voc("<pad>")

model = MBartForConditionalGeneration.from_pretrained("ai4bharat/IndicBART")
model.to(0)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.00001,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ] ## We suppose that weight decay will be used except for biases and layer norm weights.

print("Optimizing", [n for n, p in model.named_parameters() if p.requires_grad])
num_params_to_optimize = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_model_params = sum(p.numel() for p in model.parameters())
print("Number of model parameters:", num_model_params)
print("Total number of params to be optimized are: ", num_params_to_optimize)

print("Percentage of parameters to be optimized: ", 100*num_params_to_optimize/num_model_params)
    
optimizer = AdamW(optimizer_grouped_parameters, lr=0.001, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, 4000, 200) ## A warmup and decay scheduler. We use the linear scheduler for now. TODO: Enable other schedulers with a flag.

## Lets set the model to training mode
model.train()
ctr=0
for i, (input_ids, input_masks, decoder_input_ids, labels) in enumerate(generate_batches_bilingual(tokenizer, 500, is_summarization=True, batch_size=16, src_lang="hi", tgt_lang="hi", src_file_prefix="Data/combined_hindi.txt", tgt_file_prefix="Data/hindi_output.txt")):
  ctr=i
  if i%100 == 0:
    print()
    model.eval() # Set dropouts to zero
    print("Lets see how well the model is doing after ", i, "iterations of training")

    model.train() # back to training mode

  input_ids = input_ids.to(0)
  input_masks = input_masks.to(0)
  decoder_input_ids = decoder_input_ids.to(0)
  labels = labels.to(0)
  mod_compute = model(input_ids=input_ids, attention_mask=input_masks, decoder_input_ids=decoder_input_ids) ## Run the model and get logits.
  logits = mod_compute.logits
  lprobs = torch.nn.functional.log_softmax(logits, dim=-1) ## Softmax tempering of logits if needed.
  loss = label_smoothed_nll_loss(
      lprobs, labels, 0.1, ignore_index=0
  ) ## Label smoothed cross entropy loss.
  del input_ids ## Delete to avoid retention.
  del input_masks ## Delete to avoid retention.
  del decoder_input_ids ## Delete to avoid retention.
  del labels ## Delete to avoid retention.
  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  optimizer.step()
  scheduler.step()
  if i % 100 == 0:
    print("Loss for batch ", i+1, "is", round(loss.detach().cpu().numpy().item(), 2))
    print()