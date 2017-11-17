#training parameters
num_epochs = 32 
batch_size = 8 

#model parameters
embed_dim = embeddings.shape[1] #200
hidden_size = 32 # number of LSTM cells 
weight_decay = 1e-3 
learning_rate = 1e-3 

#RNN architecture
class RNN(nn.Module):
    
    def __init__(self, embed_dim, hidden_size, vocab_size, batch_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=1, batch_first=True)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        #[num_layers, batch_size, hidden_size] for (h_n, c_n)
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_size)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_size)))

    def forward(self, x_idx):
        all_x = self.embedding_layer(x_idx)   
        lstm_out, self.hidden = self.lstm(all_x.view(self.batch_size, len(x_idx), -1), self.hidden)
        h_n, c_n = self.hidden[0], self.hidden[1]
        return h_n 
        
use_gpu = torch.cuda.is_available()

model = RNN(embed_dim, hidden_size, len(word_to_idx), batch_size)
if use_gpu:
    print "found CUDA GPU..."
    model = model.cuda()
    
print model

#define loss and optimizer
criterion = nn.MarginRankingLoss(margin=1, size_average=True)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

training_loss = []
validation_loss = []

print "training..."
for epoch in range(num_epochs):
    
    running_train_loss = 0.0
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4, 
        drop_last = True)
        
    model.train()
        
    for batch in tqdm(train_data_loader):
      
        #TODO: reshape to batches in the middle or use batch_first=True!
        query_title = Variable(batch['query_title'])
        query_body = Variable(batch['query_body'])
        similar_title = Variable(batch['similar_title'])
        similar_body = Variable(batch['similar_body'])
        #TODO: for loop over random_title and random_body

        if use_gpu:
            query_title, query_body = query_title.cuda(), query_body.cuda()
            similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
            #TODO: for loop over random_title and random_body
        
        optimizer.zero_grad()
        model.hidden = model.init_hidden()

        lstm_query_title = model(query_title)
        lstm_query_body = model(query_body)
        lstm_query = (lstm_query_title + lstm_query_body)/2.0

        lstm_similar_title = model(similar_title)
        lstm_similar_body = model(similar_body)
        lstm_similar = (lstm_similar_title + lstm_similar_body)/2.0

        #TODO: for loop over random_title and random_body

        score_pos = cosine_similarity(lstm_query, lstm_similar)
        #TODO: score_neg

        loss = criterion([score_pos, score_neg], 1) #y=1
        loss.backward()
        optimizer.step()
                
        running_train_loss += loss.cpu().data[0]        
        
    #end for
    training_loss.append(running_train_loss)
    print "epoch: %4d, training loss: %.4f" %(epoch+1, running_train_loss)
    
    torch.save(model, SAVE_PATH)
#end for


"""
#generate plots
plt.figure()
plt.plot(training_loss, label='Adam')
plt.title("LSTM Model Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.savefig('./figures/training_loss.png')

plt.figure()
plt.plot(validation_loss, label='Adam')
plt.title("LSTM Model Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.savefig('./figures/validation_loss.png')
"""

        
            





