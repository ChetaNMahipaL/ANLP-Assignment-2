import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np
import torch.optim as optim
from utils import *
import pickle
from model import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


corpus_en_test = retrieve_corpus("../ted-talks-corpus/test.en")
corpus_fr_test = retrieve_corpus("../ted-talks-corpus/test.fr")

test_data_en = [remove_punctuation(word_tokenize(sentence)) for sentence in corpus_en_test]
test_data_fr = [remove_punctuation(word_tokenize(sentence)) for sentence in corpus_fr_test]

print(f"Testing data size (English): {len(test_data_en)}")
print(f"Testing data size (French): {len(test_data_fr)}")

with open('./misc/vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx_en = vocab_data['en_word']
    idx2word_en = vocab_data['en_idx']
    word2idx_fr = vocab_data['fr_word']
    idx2word_fr = vocab_data['fr_idx']

with open('./misc/hyperparameters.pkl', 'rb') as f:
    loaded_hyperparameters = pickle.load(f)

model_dim = loaded_hyperparameters['Model_dim']
num_heads = loaded_hyperparameters['Num_Head']
num_layers = loaded_hyperparameters['Num_layer']
hid_dim = loaded_hyperparameters['HID_DIM']
max_len = loaded_hyperparameters['MAX_LEN']
dropout = loaded_hyperparameters['Dropout']
learning_rate = loaded_hyperparameters['learning_rate']

model = Transformer(word2idx_en,word2idx_fr, model_dim, num_layers, num_heads,
                        hid_dim, max_len,device=device,dropout=dropout)

model.load_state_dict(torch.load('transformer.pt'))
model.to(device)

testIdx_en = prepare_data(test_data_en, word2idx_en)
testIdx_fr = prepare_data(test_data_fr, word2idx_fr)


print("Testing Begins")

dataset_test = TranslationDataset(testIdx_en, testIdx_fr)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16)

model.eval()
correct = 0
total = 0
total_loss = 0
total_tokens = 0
criterion = nn.CrossEntropyLoss(ignore_index = word2idx_fr["<pad>"])
hypotheses = []
references = []
avg_bleu_score = 0


with open('testbleu.txt', 'w') as f:
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader_test):
            context_words, target_words= batch
            context_words = context_words.to(device)
            target_words = target_words.to(device)

            sos_token_id = word2idx_en["<sos>"]
            eos_token_id = word2idx_en["<eos>"]
            outputs = model(context_words, target_words[:, :-1])

            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            target_words_out = target_words[:, 1:].contiguous().view(-1)

            loss = criterion(outputs, target_words_out)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += target_words_out.size(0)
            correct += (predicted == target_words_out).sum().item()

            generated = model.decode(context_words, 100)
            for hyp, ref in zip(generated, target_words):
                hyp = hyp.tolist()
                ref = ref.tolist()

                hyp_words = [idx2word_fr[token] for token in hyp
                            if token not in {0, sos_token_id,eos_token_id}]
                ref_words = [idx2word_fr[token] for token in ref
                            if token not in {0, sos_token_id,eos_token_id}]
            bleu_score = smoothedBleu(ref_words, hyp_words)
            f.write(f"{hyp_words}\t {bleu_score:.4f}\n")
            avg_bleu_score += bleu_score
                # print("Next")

        print(f'The avg Bleu Score is : {avg_bleu_score/len(dataloader_test)}')
        f.write(f"Averaged over all Sentences : {avg_bleu_score/len(dataloader_test):.4f}\n")

accuracy = 100 * correct / total
average_loss = total_loss / len(dataloader_test)

print(f'Test Accuracy: {accuracy:.2f}%')
print(f'Average Test Loss: {average_loss:.4f}')
