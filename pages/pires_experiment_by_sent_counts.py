import os
import time
import random
import numpy as np
import pandas as pd 
import streamlit as st 
from stqdm import stqdm
import plotly.express as px

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, default_data_collator


model_dict = {
	"mBERT": {
		"checkpoint": "bert-base-multilingual-cased",
		"special_tokens": [0, 101, 102]
		},
	"XLM-R": {
		"checkpoint": "xlm-roberta-base",
		"special_tokens": [1, 0, 2]
		}
	}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
SENTENCE_COUNT = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]

st.set_page_config(
    page_title="Streamlit App",
    layout="wide",
    initial_sidebar_state="expanded",
)


def seed_everything(seed:int=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@st.cache
def get_sent_embeddings(batch_outputs: list, batch_input_ids: list):
	# st.write(model_name)
	batch_tokens = [
			[token_embed for token_embed, token_id in zip(sent_embed, sent_ids) if token_id not in model_dict[model_name]["special_tokens"]]
		for sent_embed, sent_ids in zip(batch_outputs, batch_input_ids)
		]	

	batch_sent = [np.mean(sent_tokens, axis=0) for sent_tokens in batch_tokens]

	return batch_sent	


def extract_layer_embeddings(lang, dataloader, model, layer):
	# st.write(f"Extracting sentence embeddings for lang {lang} ...")
	layer_sent_embeddings = []

	for i, batch in enumerate(stqdm(dataloader)):
		with torch.no_grad():
			batch = {t:batch[t].to(DEVICE) for t in batch}
			model.to(DEVICE)

			outputs = model(**batch)

			# specific layer embeddings
			batch_embeddings = outputs["hidden_states"][layer]

			# numpy
			batch_embeddings = batch_embeddings.detach().cpu().numpy()
			batch_ids = batch['input_ids'].detach().cpu().numpy()

			layer_sent_embeddings.extend(get_sent_embeddings(batch_embeddings, batch_ids))	

	layer_sent_embeddings = np.array(layer_sent_embeddings)			
			
	return layer_sent_embeddings	


@st.cache
def calc_translation_vector(sentence_count, embedd_sent_1, embedd_sent_2):
	translation_v = np.mean(
		[sent_1-sent_2 for sent_1, sent_2 in zip(embedd_sent_1[:sentence_count], embedd_sent_2[:sentence_count])],
		axis=0)

	return translation_v	


def calc_accuracies(df, lang_pair, embedd_sent_1, embedd_sent_2, translation_v_dict):
	for sent_count, translation_v in stqdm(translation_v_dict.items()):
		accuracy = []
		for i, sent_2 in enumerate(stqdm(embedd_sent_2)):
			indx = find_nearest_neighbor(embedd_sent_1, sent_2, translation_v)
			accuracy.append(indx==i)
	
		accuracy = sum(accuracy)/len(accuracy)
		df.loc[len(df.index)] = [lang_pair, sent_count, accuracy] 

	return df	

def cosine_similarity(v1, v2):
    dot = np.dot(v1,v2).reshape(-1, 1)
    norma = np.linalg.norm(v1, axis=-1, keepdims=True)
    normb = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos = dot / norma / normb

    return cos


# @st.cache
def find_nearest_neighbor(embedd_sent_1: list, sent_2: np.ndarray, translation_v: np.ndarray):
	embedding_nearest = sent_2 + translation_v
	max_indx = np.argmax(cosine_similarity(embedd_sent_1, embedding_nearest), axis=0).item()

	return max_indx	


################################################### sidebar ####################################################
exp_name = st.sidebar.text_input("Experiment name, description for logging", key="inp_name", value="Pires Experiment by sentence count.")
seed = st.sidebar.number_input("Seed", key="seed_name", value=42)
model_name = st.sidebar.radio("Model", model_dict.keys())

lang_pair = st.sidebar.selectbox("wmt16", ("de-en", "cs-en", "fi-en", "ro-en", "ru-en", "tr-en"))

lang_pair_splited = lang_pair.split("-")
lang_1=lang_pair_splited[0]
lang_2=lang_pair_splited[1] 

LAYER = st.sidebar.number_input("Layer", key="layer_name", value=6)

sampling = st.sidebar.checkbox("Sampling", value=True)



@st.cache(suppress_st_warning=True, persist=True)
def run():	
	t0 = time.time()
	seed_everything(seed)

	tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name]['checkpoint'])
	model = AutoModel.from_pretrained(model_dict[model_name]['checkpoint'], output_hidden_states=True)
	

	
	# len=2169
	valid_dataset = load_dataset("wmt16", lang_pair, split="validation").shuffle(seed=seed)
	if sampling:
		valid_dataset = valid_dataset.select(range(17))
	# len=2999
	test_dataset = load_dataset("wmt16", lang_pair, split="test").shuffle(seed=seed)
	if sampling:
		test_dataset = test_dataset.select(range(17))
	# non-parallel
	other_en_dataset = load_dataset("xnli", "en", split="test").shuffle(seed=seed).select(range(len(test_dataset)))
	if sampling:
		other_en_dataset = other_en_dataset.select(range(17))

	t1 = time.time()
	raw_datasets = DatasetDict()
	raw_datasets[f"valid_{lang_1}"] = {"data": []}
	raw_datasets[f"valid_{lang_2}"] = {"data": []}
	raw_datasets[f"test_{lang_1}"] = {"data": []}
	raw_datasets[f"test_{lang_2}"] = {"data": []}
	raw_datasets[f"other_{lang_2}"] = {"data": []}

	for i, item in enumerate(test_dataset):
		raw_datasets[f"test_{lang_1}"]["data"].append(item['translation'][lang_1])
		raw_datasets[f"test_{lang_2}"]["data"].append(item['translation'][lang_2])
		raw_datasets[f"other_{lang_2}"]["data"].append(other_en_dataset[i]['hypothesis'])

	for item in valid_dataset:
		raw_datasets[f"valid_{lang_1}"]["data"].append(item['translation'][lang_1])
		raw_datasets[f"valid_{lang_2}"]["data"].append(item['translation'][lang_2])

	# convert into HF Dataset
	for k in raw_datasets:
		raw_datasets[k] = Dataset.from_dict(raw_datasets[k])	

	st.write(f"Preparing datasets ...", time.time()-t1)	

	def tokenize_dataset(examples):
		tokenized_inputs = tokenizer(
			examples['data'],
			max_length=128,
			padding='max_length',
			truncation=True
		)
		return tokenized_inputs

	t2 = time.time()
	processed_raw_datasets = raw_datasets.map(
		tokenize_dataset,
		batched=True,
		remove_columns=['data']
	)  
	st.write(f"Tokenizing ...", time.time()-t2)	

	dataloaders = {}
	for k in raw_datasets:
		dataloaders[k] = DataLoader(processed_raw_datasets[k], batch_size=64, shuffle=False, collate_fn=default_data_collator, num_workers=8)

	# shuffle parallel
	# dataloaders[f"valid_{lang_2}"] =  DataLoader(processed_raw_datasets[f"valid_{lang_2}"], batch_size=64, shuffle=True, collate_fn=default_data_collator, num_workers=4)

	t3 = time.time()
	sent_embeddings = {}
	for k in dataloaders:
		sent_embeddings[k] = extract_layer_embeddings(k, dataloaders[k], model, layer=LAYER)	

	st.write(f"Extracting embeddings ...", time.time()-t3)	

	# Calculate translation vector
	t4 = time.time()
	translation_v = {}
	translation_v_other = {}

	for s_count in stqdm(SENTENCE_COUNT):
		translation_v[s_count] = calc_translation_vector(s_count, sent_embeddings[f"valid_{lang_1}"], sent_embeddings[f"valid_{lang_2}"])
		translation_v_other[s_count] = calc_translation_vector(s_count, sent_embeddings[f"valid_{lang_1}"], sent_embeddings[f"other_{lang_2}"])

	st.write(f"Translation vector ...", time.time()-t4)	

	# Accuracies
	t5 = time.time()
	df = pd.DataFrame([], columns=["Lang Pair", "Sentence Count", "Match Accuracy"])
	
	df = calc_accuracies(df, lang_pair, sent_embeddings[f"test_{lang_1}"], sent_embeddings[f"test_{lang_2}"], translation_v)
	df = calc_accuracies(df, f"{lang_1}-non_parallel_en", sent_embeddings[f"test_{lang_1}"], sent_embeddings[f"test_{lang_2}"], translation_v_other)
	
	st.write(f"Calculating accuracies ...", time.time()-t5)	

	df['Sentence Count'] = df['Sentence Count'].values.astype(str)

	return df


################################################### chart ####################################################
if st.sidebar.button("Run Experiment") or True:
	t0 = time.time()
	df = run()

	fig = px.line(df, x="Sentence Count", y='Match Accuracy', color='Lang Pair', markers=True, title="Accuracy of the nearest neighbor translation")
	st.plotly_chart(fig, use_container_width=True)

	st.write("Overall", (time.time() - t0)/60, "min")	