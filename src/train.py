import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
from utils import *
from encoder import Encoder
from decoder import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


print("Preparing English Data Corpus")

corpus_en = retrieve_corpus("../ted-talks-corpus/train.en")
corpus_en_val = retrieve_corpus("../ted-talks-corpus/dev.en")

train_data_en = [remove_punctuation(word_tokenize(sentence)) for sentence in corpus_en]
val_data_en = [remove_punctuation(word_tokenize(sentence)) for sentence in corpus_en_val]

word_freq_en = {}
word_freq_en = create_freq(word_freq_en, train_data_en)
word_freq_en = create_freq(word_freq_en,val_data_en)

threshold = 3
train_data_en = create_unk_emb(word_freq_en,train_data_en,threshold)
val_data_en = create_unk_emb(word_freq_en,val_data_en, threshold)

vocab_en = create_vocab(word_freq_en,threshold)

word2idx_en, idx2word_en = word2idx_fun(vocab_en)

print("Preparing French Data Corpus")

corpus_fr = retrieve_corpus("../ted-talks-corpus/train.fr")
corpus_fr_val = retrieve_corpus("../ted-talks-corpus/dev.fr")

train_data_fr = [remove_punctuation(word_tokenize(sentence)) for sentence in corpus_fr]
val_data_fr = [remove_punctuation(word_tokenize(sentence)) for sentence in corpus_fr_val]

word_freq_fr = {}
word_freq_fr = create_freq(word_freq_fr, train_data_fr)
word_freq_fr = create_freq(word_freq_fr,val_data_fr)

threshold = 5
train_data_fr = create_unk_emb(word_freq_fr,train_data_fr,threshold)
val_data_fr = create_unk_emb(word_freq_fr,val_data_fr, threshold)

vocab_fr = create_vocab(word_freq_fr,threshold)

word2idx_fr, idx2word_fr = word2idx_fun(vocab_fr)

# print(len(word2idx_en))
# print(len(word2idx_fr))

#<------------------------------------------------------Word2Index------------------------------------------------------------->

Truncation = 100

print(f"Training data size (English): {len(train_data_en)}")
print(f"Training data size (French): {len(train_data_fr)}")
print(f"Validating data size (English): {len(val_data_en)}")
print(f"Validating data size (French): {len(val_data_fr)}")

trainIdx_en = prepare_data(train_data_en, word2idx_en, max_len=Truncation)
trainIdx_fr = prepare_data(train_data_fr, word2idx_fr, max_len=Truncation)
valIdx_en = prepare_data(val_data_en, word2idx_en, max_len=Truncation)
valIdx_fr = prepare_data(val_data_fr, word2idx_fr, max_len=Truncation)

print("Created input for loading")

#<-----------------------------------------------------Combining Encoder and Decoder---------------------------------------------->

class Transformer(nn.Module):
    def __init__(self, vocab_en, vocab_fr, model_dim, num_layer, num_heads, hid_dim, max_len, dropout=0.1):
        super(Transformer, self).__init__()

        self.vocab_en = vocab_en
        self.vocab_fr = vocab_fr
        self.model_dim = model_dim
        self.embed_en = nn.Embedding(len(vocab_en), model_dim)
        self.embed_fr = nn.Embedding(len(vocab_fr), model_dim)

        self.pos_enc = PosEncoding(model_dim, max_len)
        self.encoders = nn.ModuleList([Encoder(model_dim, num_heads, hid_dim, dropout) for _ in range(num_layer)])
        self.decoders = nn.ModuleList([Decoder(model_dim, num_heads, hid_dim, dropout) for _ in range(num_layer)])
        self.final_layer = nn.Linear(model_dim, len(vocab_fr))
        self.max_len = max_len

    def decode(self, src, max_len=40):
        batch_size = src.size(0)

        src_emb = self.pos_enc(self.embed_en(src))
        for encoder in self.encoders:
            src_emb = encoder(src_emb, None)

        tgt = torch.ones(batch_size, 1).long().to(device)


        for _ in range(max_len):

            src_mask, tgt_mask = self.generate_square_subsequent_mask(src, tgt)
            tgt_emb = self.pos_enc(self.embed_fr(tgt))
            for decoder in self.decoders:
                tgt_emb = decoder(tgt_emb, src_emb, src_mask, tgt_mask)

            output = self.final_layer(tgt_emb)
            next_word = output[:, -1:, :]
            next_word = torch.argmax(next_word, dim=-1)
            tgt = torch.cat((tgt, next_word), dim=1)
            # tgt = out_labels

        return tgt


    def forward(self, src, target):
        src = src
        src_mask, target_mask = self.generate_square_subsequent_mask(src, target)
        src_mask = src_mask
        target_mask = target_mask
        src_emb = self.pos_enc(self.embed_en(src))
        target_emb = self.pos_enc(self.embed_fr(target))

        for encoder in self.encoders:
            src_emb = encoder(src_emb, src_mask)

        for decoder in self.decoders:
            target_emb = decoder(target_emb, src_emb, src_mask, target_mask)

        return self.final_layer(target_emb)

    def generate_square_subsequent_mask(self,src, target):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt_mask.size(2)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(device)
        return src_mask, tgt_mask


#<----------------------------------------Preparing DataLoader---------------------------------------------------------->

dataset_train = TranslationDataset(trainIdx_en, trainIdx_fr)
dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)

dataset_val = TranslationDataset(valIdx_en, valIdx_fr)
dataLoader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=True)

#<--------------------------------------------------Model Parameters------------------------------------------------------->

Model_dim = 100
Num_Head = 4
Num_layer = 2
HID_DIM = 300
MAX_LEN = Truncation
Dropout = 0.1
learning_rate = 0.001

Model = Transformer(vocab_en=vocab_en,vocab_fr=vocab_fr, model_dim=Model_dim, num_layer=Num_layer, num_heads=Num_Head,
                        hid_dim=HID_DIM, dropout=Dropout)
Model.to(device)

CRITERION = nn.CrossEntropyLoss(ignore_index = word2idx_fr["<pad>"])
OPTIMIZER = optim.Adam(Model.parameters(), lr=learning_rate)
train_losses = []
val_losses = []

#<--------------------------------------------Training Function--------------------------------------------------------------->

def training(model, dataloader_train, dataloader_val, criterion, optimizer, num_epochs):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for _, batch in enumerate(dataloader_train):
            context_words, target_words = batch
            context_words = context_words.to(device)
            target_words = target_words.to(device)

            optimizer.zero_grad()

            outputs = model(context_words, target_words[:, :-1])

            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            target_words_out = target_words[:, 1:].contiguous().view(-1)

            loss = criterion(outputs, target_words_out)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)


        # Validation loop
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader_val:
                context_words, target_words = batch
                context_words = context_words.to(device)
                target_words = target_words.to(device)

                outputs = model(context_words, target_words[:, :-1])

                outputs = outputs.contiguous().view(-1, outputs.size(-1))
                target_words_out = target_words[:, 1:].contiguous().view(-1)

                loss = criterion(outputs, target_words_out)

                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += target_words_out.size(0)
                correct += (predicted == target_words_out).sum().item()


        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')
    
    return model

Trained_Model = training(Model, dataLoader_train, dataLoader_val, CRITERION, OPTIMIZER, num_epochs=15)

print("Training and Validation Complete.")

#<------------------------------------------- Save Model----------------------------------------------------------------------->

torch.save(Trained_Model, 'transformer.pt')
