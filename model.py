import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
import operator
from mog_lstm import MogLSTM
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from collections import Counter

from build_vocab import Vocabulary


# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',strip_accents=True)

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

class EncoderCNN(nn.Module):
    def __init__(self, target_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


class EncoderStory(nn.Module):
    def __init__(self, img_feature_size, hidden_size, n_layers):
        super(EncoderStory, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = EncoderCNN(img_feature_size)
        self.lstm = nn.LSTM(img_feature_size, hidden_size, n_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size * 2 + img_feature_size, hidden_size * 2)
        self.dropout = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(hidden_size * 2, momentum=0.01)
        self.init_weights()

    def get_params(self):
        return self.cnn.get_params() + list(self.lstm.parameters()) + list(self.linear.parameters()) + list(self.bn.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_images):
        data_size = story_images.size()
        local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4]))
        global_rnn, (hn, cn) = self.lstm(local_cnn.view(data_size[0], data_size[1], -1))
        glocal = torch.cat((local_cnn.view(data_size[0], data_size[1], -1), global_rnn), 2)
        output = self.linear(glocal)
        output = self.dropout(output)
        output = self.bn(output.contiguous().view(-1, self.hidden_size * 2)).view(data_size[0], data_size[1], -1)

        return output, (hn, cn)


class DecoderStory(nn.Module):
    def __init__(self, embed_size, hidden_size,mog_layers, vocab):
        super(DecoderStory, self).__init__()

        self.embed_size = embed_size
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = DecoderRNN(embed_size, hidden_size, mog_layers, vocab)
        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, story_feature, captions, lengths):
        story_feature = self.linear(story_feature)
        story_feature = self.dropout(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn(story_feature, captions, lengths)
        return result

    def inference(self, story_feature):
        story_feature = self.linear(story_feature)
        story_feature = F.relu(story_feature)
        result = self.rnn.inference(story_feature)
        return result


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, mog_layers, vocab,use_bert = True):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        vocab_size = len(vocab)
		if self.use_bert:
            self.embed_size = 768
        else:
            self.embed_size = embed_size
        self.dropout1 = nn.Dropout(p=0.1) 
		if not use_bert:
            self.embedding = nn.Embedding(vocab_size, self.embed_size)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # always fine-tune embeddings (even with GloVe)
            for p in self.embedding.parameters():
                p.requires_grad = True      
        # self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, n_layers, batch_first=True, dropout=0.5)
        self.lstm1 = MogLSTM(embed_size + hidden_size, hidden_size, mog_layers)
        self.lstm2 = MogLSTM(hidden_size, hidden_size, mog_layers)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, vocab_size)
		self.embed = nn.Embedding(vocab_size, self.embed_size)
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.softmax = nn.Softmax(0)

        self.brobs = []

        self.init_input = torch.zeros([5, 1, embed_size], dtype=torch.float32)

        if torch.cuda.is_available():
            self.init_input = self.init_input.cuda()

        self.start_vec = torch.zeros([1, vocab_size], dtype=torch.float32)
        self.start_vec[0][1] = 10000
        if torch.cuda.is_available():
            self.start_vec = self.start_vec.cuda()

        self.init_weights()

    def get_params(self):
        return list(self.parameters())

    def init_hidden(self):
        h0 = torch.zeros( 1, self.hidden_size)
        c0 = torch.zeros(1, self.hidden_size)
        
        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        return (h0, c0)

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0) 

    def forward(self, features, captions, lengths):
        outputs = []
        dec_len = [x-1 for x in lengths]
        max_dec_len = max(dec_len)
        caption_string = []
        for cap in captions:
            stringList = []
            for i in cap:
                word = self.vocab.idx2word[i.item()]
                if word == '<pad>' or word == '<start>' or word == '<end>' or word == '.':
                    continue
                else:
                    stringList.append(word)
            caption_string.append(stringList)
        if not self.use_bert:
            embeddings = self.embedding(captions)  #embedding
        elif self.use_bert:
            embeddings = []
            for cap_idx in  caption_string:
                
                #padd caption to correct size
                while len(cap_idx) < max_dec_len:
                    cap_idx.append('<pad>')
                    
                cap = ' '.join([words for words in cap_idx])
                encoded = cap.encode("ascii",errors = "replace")
                cap = encoded.decode("ascii")
                cap = u'[CLS] '+cap
                print(cap)
                
                tokenized_cap = tokenizer.tokenize(cap)                
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_cap)
                tokens_tensor = torch.tensor([indexed_tokens])
                

                with torch.no_grad():
                    encoded_layers, _ = model(tokens_tensor)
                    
                bert_embedding = encoded_layers.squeeze(0)
                
                split_cap = cap.split()
                tokens_embedding = []
                j = 0

                for full_token in split_cap:
                    curr_token = ''
                    x = 0
                    for i,_ in enumerate(tokenized_cap[1:]): # disregard CLS
                        token = tokenized_cap[i+j]
                        piece_embedding = bert_embedding[i+j]
                        
                        # full token
                        if token == full_token and curr_token == '' :
                            tokens_embedding.append(piece_embedding)
                            j += 1
                            break
                        else: # partial token
                            x += 1
                            
                            if curr_token == '':
                                tokens_embedding.append(piece_embedding)
                                curr_token += token.replace('#', '')
                            else:
                                tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                curr_token += token.replace('#', '')
                                
                                if curr_token == full_token: # end of partial
                                    j += x
                                    break                            
                   
                cap_embedding = torch.stack(tokens_embedding)

                embeddings.append(cap_embedding)
                
            embeddings = torch.stack(embeddings)
        if torch.cuda.is_available():
            embeddings = embeddings.cuda()
        embeddings = self.dropout1(embeddings)
        features = features.unsqueeze(1).expand(-1, np.amax(lengths), -1)
        embeddings = torch.cat((features, embeddings), 2)
       
        outputs = []
        (h1, c1) = self.init_hidden()
        (h2,c2) = self.init_hidden()

        for i, length in enumerate(lengths):
            lstm_input = embeddings[i][0:length - 1]
            output1, (h1, c1) = self.lstm1(lstm_input.unsqueeze(0), (h1, c1))
            output, (h2, c2) = self.lstm2(output1, (h2, c2))
            output = self.dropout2(output)
            output = self.linear(output[0])
            output = torch.cat((self.start_vec, output), 0)
            outputs.append(output)

        return outputs


    def inference(self, features):
        results = []
        (h1, c1) = self.init_hidden()
        (h2, c2) = self.init_hidden()
        vocab = self.vocab
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'), vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'), vocab('did')]

        cumulated_word = []
        for feature in features:

            feature = feature.unsqueeze(0).unsqueeze(0)
            predicted = torch.tensor([1], dtype=torch.long).cuda()
            lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)
            sampled_ids = [predicted,]

            count = 0
            prob_sum = 1.0

            for i in range(50):
                outputs1, (h1, c1) = self.lstm1(lstm_input, (h1, c1))
            	outputs, (h2, c2) = self.lstm2(outputs1, (h2, c2))
                outputs = self.linear(outputs.squeeze(1))

                if predicted not in termination_list:
                    outputs[0][end_vocab] = -100.0

                for forbidden in forbidden_list:
                    outputs[0][forbidden] = -100.0

                cumulated_counter = Counter()
                cumulated_counter.update(cumulated_word)

                prob_res = outputs[0]
                prob_res = self.softmax(prob_res)
                for word, cnt in cumulated_counter.items():
                    if cnt > 0 and word not in function_list:
                        prob_res[word] = prob_res[word] / (1.0 + cnt * 5.0)
                prob_res = prob_res * (1.0 / prob_res.sum())

                candidate = []
                for i in range(100):
                    index = np.random.choice(prob_res.size()[0], 1, p=prob_res.cpu().detach().numpy())[0]
                    candidate.append(index)

                counter = Counter()
                counter.update(candidate)

                sorted_candidate = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)

                predicted, _ = counter.most_common(1)[0]
                cumulated_word.append(predicted)

                predicted = torch.from_numpy(np.array([predicted])).cuda()
                sampled_ids.append(predicted)

                if predicted == 2:
                    break

                lstm_input = torch.cat((feature, self.embed(predicted).unsqueeze(1)), 2)

            results.append(sampled_ids)

        return results
