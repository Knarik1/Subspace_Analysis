import os
import time
import glob
import json
import random
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import streamlit as st 
from stqdm import stqdm
from functools import reduce
from scipy.stats import pearsonr, spearmanr
import plotly.express as px

from simalign import SentenceAligner

from datasets import load_dataset, DatasetDict, Dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, default_data_collator, AutoModel


MODEL_NAME = "xlm-roberta-base"
PATH = "/mnt/xtb/knarik/outputs/xMS/ner/xlm-roberta-base-multi-fine-tune/"
DATA_PATH = "./rd_books_kacenka/"
RESULTS_PATH = './en_order_correlations.csv'
OUTPUT_DIR = PATH + 'FineTuned_models/'
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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


def load_e3_prime_errors(model_names):
    e3_path = PATH + f'all_models_evals_en_de_zh.json'
    
    with open(e3_path, 'r+') as f:
        evals_dict = json.load(f)  
        
    # Getting only 'overall_f1' scores 
    f1_dict = {}

    for k,v in evals_dict.items():
        if k in model_names:
            f1_dict[k] = {}

            for lang_key in v: 
                f1_dict[k][lang_key] = 1 - v[lang_key]['overall_f1']    
                
                
    evals_df = pd.DataFrame(f1_dict).T
    evals_df = evals_df.rename({
        "Test_en":"e3'_en",
        "Test_de":"e3'_de",
        "Test_zh":"e3'_zh",
        "Test_es":"e3'_es",
        "Test_nl":"e3'_nl",
    }, axis=1)            
    
    return evals_df


def load_e2_prime_errors(langs):
    evals_df_list = []
    
    for lang in langs:
        e2_path = PATH + f'all_models_evals_en_de_zh_{lang}_classifier.json'

        with open(e2_path, 'r+') as f:
            evals_dict = json.load(f)    

        # Get only the last epoch f1 scores
        evals_last_epoch_dict = {model_name: {f"e2'_{lang}": 1 - item['epoch_10'][f'Test_{lang}']['overall_f1']} for model_name, item in evals_dict.items()}  
        evals_df = pd.DataFrame.from_dict(evals_last_epoch_dict).T
        evals_df_list.append(evals_df)
        
    evals_df = pd.concat(evals_df_list, axis=1) if len(evals_df_list)>1 else evals_df
    
    return evals_df   


def extract_embeddings(model, dataloaders, tokenizer):
	en_layer_sent_embeddings = []
	ce_layer_sent_embeddings = []
	
	for iter_i, (batch_en, batch_ce) in stqdm(enumerate(zip(dataloaders['en'], dataloaders['ce']))):
		with torch.no_grad():
			for i in range(len(batch_en['input_ids'])):
				en_token_ids = [id_val.item() for id_val in batch_en['input_ids'][i] if id_val not in [0,1,2]]
				ce_token_ids = [id_val.item() for id_val in batch_ce['input_ids'][i] if id_val not in [0,1,2]]
				
				en_token_labels = [order.item() for order in batch_en['token_labels'][i] if order!=-1]
				ce_token_labels = [order.item() for order in batch_ce['token_labels'][i] if order!=-1]

				en_new_token_ids = []
				ce_new_token_ids = []

				for en_ce_match in batch_en['word_match'][i]:
					if en_ce_match[0] == -1:
						break

					en_word_number = en_ce_match[0].item()
					matched_tokens_en = [j for j, val in enumerate(en_token_labels) if val==en_word_number]
					en_new_token_ids.extend([en_token_ids[token_idx] for token_idx in matched_tokens_en])

					ce_word_number = en_ce_match[1].item()
					matched_tokens_ce = [j for j, val in enumerate(ce_token_labels) if val==ce_word_number]
					ce_new_token_ids.extend([ce_token_ids[token_idx] for token_idx in matched_tokens_ce])
				
				# if sent len > 126, truncate
				en_new_token_ids = en_new_token_ids if len(en_new_token_ids)<=126 else en_new_token_ids[:126]
				ce_new_token_ids = ce_new_token_ids if len(ce_new_token_ids)<=126 else ce_new_token_ids[:126]

				batch_en['input_ids'][i] = torch.tensor([0] + en_new_token_ids + [2] + [1]*(128-len(en_new_token_ids)-2))
				batch_ce['input_ids'][i] = torch.tensor([0] + ce_new_token_ids + [2] + [1]*(128-len(ce_new_token_ids)-2))


			batch_en_input = {t:batch_en[t].to(DEVICE) for t in batch_en if t in ['attention_mask', 'input_ids']}
			batch_ce_input = {t:batch_ce[t].to(DEVICE) for t in batch_ce if t in ['attention_mask', 'input_ids']}

			en_outputs = model(**batch_en_input) 
			ce_outputs = model(**batch_ce_input) 
			
			# specific layer embeddings
			en_batch_embeddings = en_outputs["hidden_states"][-1]
			ce_batch_embeddings = ce_outputs["hidden_states"][-1]

			# numpy
			en_batch_embeddings = en_batch_embeddings.detach().cpu().numpy()
			ce_batch_embeddings = ce_batch_embeddings.detach().cpu().numpy()
			en_batch_ids = batch_en['input_ids'].detach().cpu().numpy()
			ce_batch_ids = batch_ce['input_ids'].detach().cpu().numpy()

			en_layer_sent_embeddings.extend(get_sent_embeddings(en_batch_embeddings, en_batch_ids))	
			ce_layer_sent_embeddings.extend(get_sent_embeddings(ce_batch_embeddings, ce_batch_ids))	
			
	en_layer_sent_embeddings = np.array(en_layer_sent_embeddings)				
	ce_layer_sent_embeddings = np.array(ce_layer_sent_embeddings)			
			
	return en_layer_sent_embeddings, ce_layer_sent_embeddings  


@st.cache
def get_sent_embeddings(batch_outputs: list, batch_input_ids: list):
    # st.write(model_name)
    batch_tokens = [
            [token_embed for token_embed, token_id in zip(sent_embed, sent_ids) if token_id not in [0,1,2]]
        for sent_embed, sent_ids in zip(batch_outputs, batch_input_ids)
        ]	

    batch_sent = [np.mean(sent_tokens, axis=0) for sent_tokens in batch_tokens]

    return batch_sent      


def cosine_similarity(v1, v2):
    # elementwise then take sum by rows
    dot = np.sum(v1 * v2, axis=1, keepdims=True)
    norma = np.linalg.norm(v1, axis=-1, keepdims=True)   
    normb = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos = dot / norma / normb
    avg_cos = np.mean(cos).item()

    return avg_cos            


exp_name = st.sidebar.text_input("Experiment", key="exp_name", value="Target language has the same order as English")
seed = st.sidebar.number_input("Bootstrapping seed", key="seed", value=42)


@st.cache(suppress_st_warning=True)
def run():	
    seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, output_hidden_states=True, num_labels=9)
    model.to(DEVICE)
    model.eval()

    # Loading Errors
    # MODEL_PATHS = random.sample(sorted(glob.glob(os.path.join(OUTPUT_DIR, f'model-seed_*')))[:100], 100)
    MODEL_PATHS = glob.glob(os.path.join(OUTPUT_DIR, f'model-seed_*'))[:100]
    model_names = [path.split('/')[-1] for path in MODEL_PATHS]

    if os.path.isfile(RESULTS_PATH):
        eval_eval = pd.read_csv(RESULTS_PATH)

        return eval_eval 

    eval_e3_prime_df = load_e3_prime_errors(model_names)
    eval_e2_prime_df = load_e2_prime_errors(['es', 'nl'])

    eval_e3_prime_df["Dev_en_de_zh"] = (eval_e3_prime_df["Dev_en"] + eval_e3_prime_df["Dev_de"] + eval_e3_prime_df["Dev_zh"])/3
    eval_e3_prime_df.sort_values("e3'_en")
    
    eval_df = pd.concat((eval_e2_prime_df, eval_e3_prime_df), axis=1)
    eval_df.sort_values("Dev_en", ascending=True)

    eval_df['e3_es'] = eval_df["e3'_es"] - eval_df["e2'_es"] 
    eval_df['e3_nl'] = eval_df["e3'_nl"] - eval_df["e2'_nl"]

    # Data Preparation
    t1 = time.time()
    data_files = glob.glob(os.path.join(DATA_PATH, f'*.wa'))
    # 20 files each 50 sentence
    data_pd = pd.concat([pd.read_xml(data_file) for data_file in data_files], axis=0, ignore_index=True)
 
    st.write(data_pd)  

    raw_datasets = DatasetDict()
    raw_datasets["en"] = {"data": [], "word_match": []}
    raw_datasets["ce"] = {"data": [], "word_match": []}

    for i in stqdm(range(len(data_pd))):
        raw_datasets["en"]["data"].append(data_pd.loc[i]['english'])
        raw_datasets["ce"]["data"].append(data_pd.loc[i]['czech'])

        word_match = data_pd.loc[i]['sure'].split()
        word_match = [[int(pair.split("-")[0])-1, int(pair.split("-")[1])-1] for pair in word_match]

        raw_datasets["en"]["word_match"].append(word_match)
        raw_datasets["ce"]["word_match"].append(word_match)

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

        token_labels = []
        for i, label in enumerate(examples['word_match']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            token_labels.append(list(map(lambda a: -1 if a is None else a, word_ids)))
        tokenized_inputs['token_labels'] = token_labels  
        tokenized_inputs['word_match'] = [sent_word_match + [[-1,-1]]*(128-len(sent_word_match)) for sent_word_match in examples['word_match']]  

        return tokenized_inputs

    t2 = time.time()
    processed_raw_datasets = raw_datasets.map(
        tokenize_dataset,
        batched=True,
        remove_columns=['data']
    ) 

    st.write(f"Tokenizing ...", time.time()-t2)	 

    dataloaders = {}
    for k in processed_raw_datasets:
        dataloaders[k] = DataLoader(processed_raw_datasets[k], batch_size=64, shuffle=False, collate_fn=default_data_collator, num_workers=8)

    t3 = time.time()
    model_clf_dict = {
        "embedd_sim": {}
    }

    st.write("Extracting sentence embeddings ...")
    for model_path in stqdm(MODEL_PATHS):
        model_name = model_path.split('/')[-1]
        model.load_state_dict(torch.load(model_path+"/pytorch_model.bin"))
        model.to(DEVICE)
        model.eval()

        en_sent_embedds, ce_sent_embedds = extract_embeddings(model, dataloaders, tokenizer)
        embedd_sim = cosine_similarity(en_sent_embedds, ce_sent_embedds)	
        model_clf_dict["embedd_sim"][model_name] = embedd_sim

    st.write(f"Extracting embeddings ...", time.time()-t3)

    embedd_sim_df = pd.DataFrame(model_clf_dict) 
    eval_df = eval_df.join(embedd_sim_df)

    return eval_df


################################################### chart ####################################################
if st.sidebar.button("Run Experiment") or True:
    t0 = time.time()
    eval_df = run()

    # Remove outliers
    eval_df = eval_df.sort_values("Dev_en_de_zh", ascending=True).head(94) 
    MODEL_NAMES = eval_df.index 
    st.header("English-Czech word aligned parallel data.") 
    st.write(eval_df)
    st.write("Overall", (time.time() - t0)/60, "min")	

    # show top 5 
    st.write("Ordered by embedd_sim column")
    st.write(eval_df.sort_values("embedd_sim", ascending=False)[["embedd_sim", "Dev_en_de_zh", "e2'_es", "e2'_nl", "e3'_es", "e3'_nl"]][:5])
    st.write("Ordered by Dev_en_de_zh column")
    st.write(eval_df.sort_values("Dev_en_de_zh", ascending=True)[["Dev_en_de_zh", "embedd_sim", "e2'_es", "e2'_nl", "e3'_es", "e3'_nl"]][:5])

    for lang in ["es", "nl"]:
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4"))
        subplot_titles = []

        for row, x_axis_name in enumerate(["Dev_en_de_zh", "embedd_sim"]):
            for col, y_axis_name in enumerate([f"e3'_{lang}", f"e2'_{lang}"]):
                fig.add_trace(
                    go.Scatter(x=eval_df[x_axis_name], y=eval_df[y_axis_name], mode='markers', name=f"fine-tuned model"),
                    row=row+1,
                    col=col+1
                )
                fig.update_xaxes(title_text=x_axis_name, row=row+1, col=col+1)
                fig.update_yaxes(title_text=y_axis_name, row=row+1, col=col+1)

                # correlations
                corr_s = spearmanr(eval_df[x_axis_name], eval_df[y_axis_name]).correlation
                corr_p = pearsonr(eval_df[x_axis_name], eval_df[y_axis_name]).statistic
                subplot_titles.append(f"Spearman={corr_s:.2f}, Pearson={corr_p:.2f}")


        # update subtitles
        for i, title_text in enumerate(subplot_titles):
            fig.layout.annotations[i].update(text=title_text)

        fig.update_layout(title_text=f"Correlations for {lang}", height=600, width=600)
        # fig.update_yaxes(
        #     scaleanchor = "x",
        #     scaleratio = 0.5,
        # )
        st.plotly_chart(fig, use_container_width=True)


        # fig 1
        corr_s = spearmanr(eval_df["Dev_en_de_zh"], eval_df["embedd_sim"]).correlation
        corr_p = pearsonr(eval_df["Dev_en_de_zh"], eval_df["embedd_sim"]).statistic

        fig_1 = px.scatter(eval_df, x='Dev_en_de_zh', y='embedd_sim', title=f"Spearman={corr_s:.2f}, Pearson={corr_p:.2}")  
        st.plotly_chart(fig_1, use_container_width=True)  



    # Bootstrapping
    st.header("Bootstrapping Approach #1")
    bootstraped_results_df = pd.DataFrame()
    bootstraped_df = pd.DataFrame(columns=["Dev_en_de_zh", "embedd_sim"])
    
    for lang in stqdm(["es", "nl"]):
        for err_column in [f"e3'_{lang}"]:
            for i in stqdm(range(1000)):
                model_names_bootstraped = np.random.choice(MODEL_NAMES, 50)
                sampled_df_rand = eval_df.loc[model_names_bootstraped]

                bootstraped_dict = {}
                
                bootstraped_dict["Dev_en_de_zh"] = sampled_df_rand[["Dev_en_de_zh", err_column]].sort_values("Dev_en_de_zh", ascending=True).head(1)[err_column].item()
                bootstraped_dict["embedd_sim"] = sampled_df_rand[["embedd_sim", err_column]].sort_values("embedd_sim", ascending=False).head(1)[err_column].item()
            
                bootstraped_df.loc[i] = bootstraped_dict

            min_cols = []

            for index, row in bootstraped_df.iterrows():
                min_cols.extend(row[row == row.min()].index.values)   

            win_col_df = pd.DataFrame({"win_col": min_cols})
            bootstraped_results_df[err_column] = win_col_df.value_counts().sort_index(ascending=False)

    st.dataframe(bootstraped_results_df.style.highlight_max(color = 'lightgreen', axis = 0))   


    st.header("Bootstrapping Approach #2") 
    bootstraped_results_df_all  = []
    for lang in stqdm(["es", "nl"]):
        bootstraped_results_df_2 = pd.DataFrame(columns=[f"e3'_{lang}"])

        for err_column in [f"e3'_{lang}"]:

            dev_diff = []
            embedd_diff = []

            for i in stqdm(range(1000)):
                model_names_bootstraped = np.random.choice(MODEL_NAMES, 50)
                sampled_df_rand = eval_df.loc[model_names_bootstraped]   

                dev_err = sampled_df_rand[["Dev_en_de_zh", err_column]].sort_values("Dev_en_de_zh", ascending=True).head(1)[err_column].item()
                embedd_err = sampled_df_rand[["embedd_sim", err_column]].sort_values("embedd_sim", ascending=False).head(1)[err_column].item()
                min_err = sampled_df_rand[[err_column]].sort_values(err_column, ascending=True).min().values[0]

                dev_diff.append(dev_err-min_err)
                embedd_diff.append(embedd_err-min_err)
            
            bootstraped_results_df_2[err_column] = pd.DataFrame({
                "Embedd_mean_err_diff": [sum(embedd_diff)/len(embedd_diff)],
                "Dev_mean_err_diff": [sum(dev_diff)/len(dev_diff)]
                }).T

        bootstraped_results_df_2 = bootstraped_results_df_2.round(4)
        bootstraped_results_df_all.append(bootstraped_results_df_2)

    st.dataframe(pd.concat(bootstraped_results_df_all, axis=1).style.highlight_min(color = 'lightgreen', axis = 0))      
                








            
