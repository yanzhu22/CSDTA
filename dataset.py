from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import joblib
from transformers import TrainingArguments
from sklearn import preprocessing
import sentencepiece as spm

sp_prot = spm.SentencePieceProcessor()
sp_prot.Load("./CSDTA/protein_sent_token.model")
 
    
sp_smi = spm.SentencePieceProcessor()
sp_smi.Load("./CSDTA/chembel_smi_sent_token.model")



CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)
PT_FEATURE_SIZE = 40
CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 }

CHARPROTLEN = 25

def label_sequence(line, MAX_SEQ_LEN):
    X = np.zeros(MAX_SEQ_LEN, dtype=np.int)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = CHARPROTSET[ch]
    return X

def label_smiles(line, max_smi_len):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] 

    return X

    

class SeqEmbDataset(Dataset):
    def __init__(self, data_path, prots, drugs ,phase, max_seq_len, max_smi_len):
        data_path = Path(data_path)

        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        ligands_df = pd.read_csv(data_path / f"{phase}_smi_can.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        ligands_can = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands
        self.max_smi_len = max_smi_len
        self.smi_can = ligands_can
        
        self.prots = prots
        self.drugs = drugs
        self.max_seq_len = max_seq_len
        self.pdbids = ligands_df['pdbid'].values

        self.length = len(self.smi)
        self.phase = phase
        prot_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
        prots_seq = {i["id"]: i["seq"] for _, i in prot_df.iterrows()}
        self.prots_seq = prots_seq

    def __getitem__(self, idx):
        pdb_id = self.pdbids[idx]

        smile =   self.smi[pdb_id]
        protseq = np.zeros((self.max_seq_len, 1024), dtype=np.float32)
        protseq[:self.prots[pdb_id].shape[0]] = self.prots[pdb_id]
        smile_emb = np.zeros((self.max_smi_len, 384), dtype=np.float32)
        drug_bert_emb = self.drugs[pdb_id]
        if drug_bert_emb.shape[0] >  self.max_smi_len:
            smile_emb[:self.max_smi_len] = drug_bert_emb[:self.max_smi_len]
        else:
            smile_emb[:drug_bert_emb.shape[0]] = drug_bert_emb

        return ( protseq,
                smile_emb,
                label_sequence( self.prots_seq[pdb_id],self.max_seq_len),
                label_smiles(smile, self.max_smi_len),
                np.array(self.affinity[pdb_id], dtype=np.float32))
    def __len__(self):
        return self.length
