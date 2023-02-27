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


st.set_page_config(
    page_title="Streamlit App",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
def get_dataset(pair: str):
	dataset = load_dataset("wmt16", pair, split="test")
	return dataset


@st.cache
def get_another_dataset(lang: str):
	dataset = load_dataset("xnli", lang, split="test")
	return dataset	


@st.cache
def get_sent_embeddings(batch_outputs: list, batch_input_ids: list):
	batch_tokens = [
			[token_embed.numpy() for token_embed, token_id in zip(sent_embed, sent_ids) if token_id not in model_dict[model_name]["special_tokens"]]
		for sent_embed, sent_ids in zip(batch_outputs, batch_input_ids)
		]	

	batch_sent = [np.mean(sent_tokens, axis=0) for sent_tokens in batch_tokens]

	return batch_sent


@st.cache
def find_nearest_neighbor(embeddings_1: list, v2: np.ndarray, translation_v: np.ndarray):
	embeddings_1 = np.array(embeddings_1)
	embedding_nearest = v2 + translation_v
	dists = np.linalg.norm(embeddings_1 - embedding_nearest, axis=1)
	min_indx = np.argmin(dists)

	return min_indx


def calc_accuracies(df, lang_pair, all_layers_sent_1, all_layers_sent_2, all_layers_translation_v, layer=None, sent_count=None):
	# @TODO -> reduce
	if layer is None:
		for l in stqdm(range(13)):
			accuracy = []

			for i, sent_2 in enumerate(all_layers_sent_2[l]):
				indx = find_nearest_neighbor(all_layers_sent_1[l], sent_2, all_layers_translation_v[l])

				accuracy.append(indx==i)
			
			accuracy = sum(accuracy)/len(accuracy)
			df.loc[len(df.index)] = [l, lang_pair, sent_count, accuracy] 


	else: 
		accuracy = []

		for i, sent_2 in enumerate(all_layers_sent_2[layer]):
			indx = find_nearest_neighbor(all_layers_sent_1[layer], sent_2, all_layers_translation_v[layer])

			accuracy.append(indx==i)
		
		accuracy = sum(accuracy)/len(accuracy)
		df.loc[len(df.index)] = [layer, lang_pair, sent_count, accuracy] 

	return df	


def extract_layer_embeddings(lang, dataloader, model, layer):
	st.write(f"Extracting sentence embeddings for lang {lang} ...")
	all_layers_sent_embeddings = {l:[] for l in range(13)}

	for i, batch in enumerate(stqdm(dataloader)):
		with torch.no_grad():
			outputs = model(**batch)

			if layer:
				# specific layer embeddings
				batch_embeddings = outputs["hidden_states"][layer]
				all_layers_sent_embeddings[layer].extend(get_sent_embeddings(batch_embeddings, batch['input_ids']))	
			else:	
				# all layers embeddings
				batch_embeddings_for_layers = outputs["hidden_states"]
				
				for l, batch_embeddings in enumerate(batch_embeddings_for_layers):
					# get only token embeddings
					all_layers_sent_embeddings[l].extend(get_sent_embeddings(batch_embeddings, batch['input_ids']))

	return all_layers_sent_embeddings


@st.cache
def calc_translation_vector(sentence_count, all_layers_sent_1, all_layers_sent_2):
	all_layers_translation_v = {}

	for l in range(13):
		all_layers_translation_v[l] = np.mean(
			[sent_1-sent_2 for sent_1, sent_2 in zip(all_layers_sent_1[l][:sentence_count], all_layers_sent_2[l][:sentence_count])],
			axis=0)

	return all_layers_translation_v		





################################################### sidebar ####################################################
exp_name = st.sidebar.text_input("Experiment name, description for logging", key="inp_name", value="Pires Experiment by layers.")
seed = st.sidebar.number_input("Seed", key="seed_name", value=42)
model_name = st.sidebar.radio("Model", model_dict.keys())

lang_pair = st.sidebar.selectbox("wmt16", ("de-en", "cs-en", "fi-en", "ro-en", "ru-en", "tr-en"))

lang_pair_splited = lang_pair.split("-")
lang_1=lang_pair_splited[0]
lang_2=lang_pair_splited[1]

x_axis = st.sidebar.radio("X axis", ['Layer', 'Sentence Count'])

SENTENCE_COUNT = [st.sidebar.slider("Number of sentences?", min_value=1, max_value=1000, value=55)]

################################################### chart ####################################################
if st.sidebar.button("Run Experiment"):
	seed_everything(seed)
	
	tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name]['checkpoint'])
	model = AutoModel.from_pretrained(model_dict[model_name]['checkpoint'], output_hidden_states=True)
	model.eval()

	# Defining DataFrame
	df = pd.DataFrame([], columns=["Layer", "Lang Pair", "Sentence Count", "Match Accuracy"])

	# 2 datasets
	translation_dataset = get_dataset(lang_pair)
	other_dataset = get_another_dataset("en")


	# sample
	translation_dataset = translation_dataset.shuffle(seed=seed)
	other_dataset = other_dataset.shuffle(seed=seed).select(range(len(translation_dataset)))

	raw_datasets = DatasetDict()

	dataset_1_raw = {"data": []}
	dataset_2_raw = {"data": []}
	dataset_other_raw = {"data": []}

	for i, item in enumerate(translation_dataset):
		dataset_1_raw["data"].append(item['translation'][lang_1])
		dataset_2_raw["data"].append(item['translation'][lang_2])
		dataset_other_raw["data"].append(other_dataset[i]['premise'])

	raw_datasets[lang_1] = Dataset.from_dict(dataset_1_raw)
	raw_datasets[lang_2] = Dataset.from_dict(dataset_2_raw)
	raw_datasets["other"] = Dataset.from_dict(dataset_other_raw)

	def tokenize_dataset(examples):
		tokenized_inputs = tokenizer(
			examples['data'],
			max_length=128,
			padding='max_length',
			truncation=True
		)
		return tokenized_inputs

	processed_raw_datasets = raw_datasets.map(
		tokenize_dataset,
		batched=True,
		remove_columns=['data']
	)  

	dataloader_1 = DataLoader(processed_raw_datasets[lang_1], batch_size=16, shuffle=False, collate_fn=default_data_collator)
	dataloader_2 = DataLoader(processed_raw_datasets[lang_2], batch_size=16, shuffle=False, collate_fn=default_data_collator)		
	dataloader_other = DataLoader(processed_raw_datasets["other"], batch_size=16, shuffle=False, collate_fn=default_data_collator)		

			
	# lang 1 dataset
	all_layers_sent_1 = extract_layer_embeddings(lang_1, dataloader_1, model, layer=LAYER)

	# lang 2 dataset
	all_layers_sent_2 = extract_layer_embeddings(lang_2, dataloader_2, model, layer=LAYER)

	# other dataset in en
	all_layers_sent_other = extract_layer_embeddings("other", dataloader_other, model, layer=LAYER)

	st.write(f"Calculating nearest neighbor accuracies ...")
	for s_count in stqdm(SENTENCE_COUNT):
		# Calculate translation vector for each layer v1 <-> v2
		all_layers_translation_v = calc_translation_vector(s_count, all_layers_sent_1, all_layers_sent_2)
		all_layers_translation_v_other = calc_translation_vector(s_count, all_layers_sent_1, all_layers_sent_other)


		# Accuracies
		df = calc_accuracies(df, lang_pair, all_layers_sent_1, all_layers_sent_2, all_layers_translation_v, layer=LAYER, sent_count=s_count)
		df = calc_accuracies(df, f"{lang_1}-non_parallel_en", all_layers_sent_1, all_layers_sent_2, all_layers_translation_v_other, layer=LAYER, sent_count=s_count)
	
	df['Sentence Count'] = df['Sentence Count'].values.astype(str)

	fig = px.line(df, x=x_axis, y='Match Accuracy', color='Lang Pair', markers=True, title="Accuracy of the nearest neighbor translation")
	st.plotly_chart(fig, use_container_width=True)
