# import necessary libraries for an LSTM model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

# import other required libraries
from io import open
from conllu import parse
import random
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')

# use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data from conllu files
dev_conllu_data = parse(open("en_atis-ud-dev.conllu", "r", encoding="utf-8").read())
train_conllu_data = parse(open("en_atis-ud-train.conllu", "r", encoding="utf-8").read())
test_conllu_data = parse(open("en_atis-ud-test.conllu", "r", encoding="utf-8").read())

# create dictionaries for each dataset
# each dictionary has the sentence id as the key and a list of information as the value
def create_data_dict(conllu_data):
    data_dict = {}
    for sentence in conllu_data:
        sent_id = sentence.metadata['sent_id']
        word_info_list = []
        for token in sentence:
            word_info = [token['id'], token['form'], token['upos']]
            word_info_list.append(word_info)
        data_dict[sent_id] = word_info_list
    return data_dict

dev_data_dict = create_data_dict(dev_conllu_data)
train_data_dict = create_data_dict(train_conllu_data)
test_data_dict = create_data_dict(test_conllu_data)

combined_dict = dev_data_dict.copy()
combined_dict.update(train_data_dict)
combined_dict.update(test_data_dict)
combined_dict

# create a vocabulary of words and tags by iterating over all 3 datasets
def return_vocab(index, extras):  
    vocab = set()  
    vocab.update(extras)
    for _, word_info_list in train_data_dict.items():
        for word_info in word_info_list:
            word_form = word_info[index]
            vocab.add(word_form)
    return vocab 

vocab = return_vocab(1, set(['<pad>','<unk>','<s>','</s>']))
pos_vocab = return_vocab(2, set(['PAD', 'START', 'END','SYM']))
(pos_vocab)

# create dictionaries to map words and tags to indices
# and vice-versa
index_to_word = {i: list(vocab)[i] for i in range(len(vocab)) }
word_to_index = {word: i for i, word in index_to_word.items()}
index_to_tag = {i: list(pos_vocab)[i] for i in range(len(pos_vocab))}
tag_to_index = {tag: i for i, tag in index_to_tag.items()}

# finding the maximum sentence length
# need this for padding, because all sentences in a batch need to be of the same length
longest_sentence_key = max(combined_dict, key = lambda k: len(combined_dict[k]))
max_sent_len = len(combined_dict[longest_sentence_key])

class LSTMTagger(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_tags):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)

    def forward(self, sentence):
        embedded = self.embedding(sentence)
        output, _ = self.lstm(embedded)
        tag_space = self.hidden2tag(output)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores.permute(0,2,1)
    
# hyperparameters and other variables
vocab_size = len(vocab)
num_tags = len(pos_vocab)
embedding_dim = 64
hidden_dim = 128
num_layers = 1
num_epochs = 10

def train_new_model(model_name):
    
    default = input("Do you want to use the default hyperparameters? (y/n): ")
    if default == 'n':
        embedding_dim = int(input("Enter embedding dimension: "))
        hidden_dim = int(input("Enter hidden dimension: "))
        num_layers = int(input("Enter number of layers: "))
        num_epochs = int(input("Enter number of epochs: "))
    else:
        embedding_dim = 64
        hidden_dim = 128
        num_layers = 1
        num_epochs = 10
        print("Using default hyperparameters")
        print(f"                 Embedding dimension:\t{embedding_dim}\n \
                Hidden dimension:\t{hidden_dim}\n \
                Number of layers:\t{num_layers}\n \
                Number of epochs:\t{num_epochs}")

    # create lists of sentences and tags for training, validation and testing
    train_sent_words = []
    train_sent_tags = []
    for word_info_list in train_data_dict.values():
        word_list = [x[1] for x in word_info_list]
        tag_list = [x[2] for x in word_info_list]
        sent = ['<s>']+word_list+['</s>'] + ['<pad>']*(max_sent_len-len(word_list))
        tags = ['START']+tag_list+['END'] + ['PAD']*(max_sent_len-len(tag_list))
        train_sent_words.append(sent)
        train_sent_tags.append(tags)

    val_sent_words = []
    val_sent_tags = []
    for word_info_list in dev_data_dict.values():
        word_list = [x[1] if x[1] in vocab else '<unk>' for x in word_info_list]
        tag_list = [x[2] for x in word_info_list]
        sent = ['<s>']+word_list+['</s>'] + ['<pad>']*(max_sent_len-len(word_list))
        tags = ['START']+tag_list+['END'] + ['PAD']*(max_sent_len-len(tag_list))
        val_sent_words.append(sent)
        val_sent_tags.append(tags)

    test_sent_words = []
    test_sent_tags = []
    for word_info_list in test_data_dict.values():
        word_list = [x[1] if x[1] in vocab else '<unk>' for x in word_info_list]
        tag_list = [x[2] for x in word_info_list]
        sent = ['<s>']+word_list+['</s>'] + ['<pad>']*(max_sent_len-len(word_list))
        tags = ['START']+tag_list+['END'] + ['PAD']*(max_sent_len-len(tag_list))
        test_sent_words.append(sent)
        test_sent_tags.append(tags)

    # add some '<unk>'s randomly
    def add_random_unks(sentences):
        for i, sentence in enumerate(sentences):
            n = len(sentence)
            num_replace = int(0.05 * n)
            replace_indices = random.sample(range(n), num_replace)
            for j in replace_indices:
                sentences[i][j] = '<unk>'
        return sentences

    train_sent_words = add_random_unks(train_sent_words)

    batch_size = 32

    # Convert the data from strings to tensors
    train_data_numerical = [[word_to_index[word] for word in sent] for sent in train_sent_words]
    train_labels = [[tag_to_index[tag] for tag in sent] for sent in train_sent_tags]
    train_data = TensorDataset(torch.LongTensor(
        train_data_numerical), torch.LongTensor(train_labels))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # batches the data and returns an iterable

    dev_data = [[word_to_index[word] for word in sent] for sent in val_sent_words]
    dev_labels = [[tag_to_index[tag] for tag in sent] for sent in val_sent_tags]
    validation_data = TensorDataset(torch.LongTensor(
        dev_data), torch.LongTensor(dev_labels))
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=True)

    test_data_numerical = [[word_to_index[word] for word in sent] for sent in test_sent_words]
    test_labels = [[tag_to_index[tag] for tag in sent] for sent in test_sent_tags]
    test_data = TensorDataset(torch.LongTensor(
        test_data_numerical), torch.LongTensor(test_labels))
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Build the LSTM model

    model = LSTMTagger(vocab_size, embedding_dim, hidden_dim, num_layers, num_tags)
    model = model.to(device)

    # Train the model
    loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_index['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epoch_idx = 0
    print("Training model...\n")
    print("Epoch \t Train Loss \t Val Loss\n=================================")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        count = 0
        for batch_idx, data in enumerate(train_loader):
            model.zero_grad()
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            targets = data[1]
            targets = targets.to(device)
            output = model(data[0])
            output = output.to(device)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            output = torch.argmax(output,dim=1)
            train_loss += loss
            count += 1
        train_loss /= count
        epoch_idx += 1
        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader):
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                for sent_idx in range(len(data[0])):
                    for word_idx in range(len(data[0][sent_idx])):
                        word = index_to_word[data[0][sent_idx][word_idx].item()]
                        if (word in vocab) == False:
                            data[0][sent_idx][word_idx] = '<unk>'
                output = model(data[0])
                targets = data[1]
                targets = targets.to(device)
                loss = loss_fn(output, targets)
                val_loss += loss
                count += 1
                targets = torch.argmax(targets, axis=1)

            val_loss /= count
            print(f'{epoch_idx}\t', end='')
            print('{:.6f}'.format(train_loss), end='\t')
            print('{:.6f}'.format(val_loss))

    # Evaluate the model
    correct = 0
    total = 0
    model.eval()
    test_loss = 0
    precision = 0
    recall = 0
    fscore = 0
    accuracy = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            for sent_idx in range(len(data[0])):
                for word_idx in range(len(data[0][sent_idx])):
                    word = index_to_word[data[0][sent_idx][word_idx].item()]
                    if (word in vocab) == False:
                        data[0][sent_idx][word_idx] = '<unk>'
            targets = data[1]
            targets = targets.to(device)
            output = model(data[0])
            indices = torch.argmax(output,dim=1)
            for i in range(len(data[0])):
                report = classification_report(targets[i].cpu(),indices[i].cpu(), 
                                                output_dict=True, zero_division=0)
                precision += report["weighted avg"]["precision"]
                recall += report["weighted avg"]["recall"]
                fscore += report["weighted avg"]["f1-score"]
                accuracy += report["accuracy"]
            loss = loss_fn(output, targets)
            test_loss += loss
    test_loss /= len(test_loader.dataset)
    precision /= len(test_loader.dataset)
    recall /= len(test_loader.dataset)
    fscore /= len(test_loader.dataset)
    accuracy /= len(test_loader.dataset)
    print('\nTest Loss: {:.6f}'.format(test_loss))
    print(f"\nClassification report:\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {fscore}")

    if model_name != "pdf.file":
        torch.save({'word_to_index': word_to_index,
                    'tag_to_index': tag_to_index,
                    'vocab_size': vocab_size,
                    'num_layers': num_layers,
                    'num_tags': num_tags,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict' : optimizer.state_dict(),
                    'hidden_dim' : model.hidden_dim,
                    'embedding_dim' : embedding_dim,
                    'index_to_tag' :index_to_tag,
                    }, model_name)
    else:
        print("\nYour model has been trained :D")
        use_model = input("Do you want to try it out? (y/n): ")
        while use_model == 'y':
            input_sent = input("Enter a sentence: ")
            tokens = word_tokenize(input_sent)
            words = ['<s>'] + tokens + ['</s>'] + ['<pad>'] * \
                        (max_sent_len - len(tokens))
            input_num = [[word_to_index[word] if word in vocab else 
                                word_to_index['<unk>'] for word in words]]
            data = torch.LongTensor(input_num)

            with torch.no_grad():
                output = model(data.to(device))
                output = torch.argmax(output, dim=1)

            output = [index_to_tag[idx.item()] for idx in output[0]]
            print('\nWORD\tTAG')
            for i in range(len(tokens)):
                print(f'{words[i+1]}\t{output[i+1]}')
            
            input_sent = input("You can enter another sentence or type 'exit' to quit:\n")
            if input_sent == 'exit':
                break  
    

def use_my_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    word_to_index = checkpoint['word_to_index']
    tag_to_index = checkpoint['tag_to_index']
    index_to_tag = checkpoint['index_to_tag']
    vocab_size = checkpoint['vocab_size']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    embedding_dim = checkpoint['embedding_dim']
    num_tags = checkpoint['num_tags']
    loaded_model = LSTMTagger(vocab_size, embedding_dim, hidden_dim, num_layers, num_tags)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    input_sent = input("Enter a sentence: ")
    while True:
        tokens = word_tokenize(input_sent)
        words = ['<s>'] + tokens + ['</s>'] + ['<pad>'] * \
                    (max_sent_len - len(tokens))
        input_num = [[word_to_index[word] if word in vocab else 
                            word_to_index['<unk>'] for word in words]]
        data = torch.LongTensor(input_num)

        with torch.no_grad():
            output = loaded_model(data.to(device))
            output = torch.argmax(output, dim=1)

        output = [index_to_tag[idx.item()] for idx in output[0]]
        print('\nWORD\tTAG')
        for i in range(len(tokens)):
            print(f'{words[i+1]}\t{output[i+1]}')
        
        input_sent = input("\nYou can enter another sentence or type 'exit' to quit:\n")
        if input_sent == 'exit':
            break
        
res = input("\nDo you want to load a pretrained model? (y/n): ")
if res == "y":
    model = input("Enter the name of the model you want to load: ")
    print("The model has been loaded")
    use_my_model(model)
elif res == "n":
    save = input("We will train a new model for you. Would you like to save it? (y/n): ")
    if save == "y":
        saved_model = input("What name would you like to save your model with? ")
        train_new_model(saved_model)        
        print("\nTrained a new model and saved it to", saved_model)
    else:
        train_new_model("pdf.file")
    if save == "y":
        use_model = input("Do you want to try it out? (y/n) ")
        if use_model == "y":
            use_my_model(saved_model)
        elif use_model == "n":
            print("Exiting...")