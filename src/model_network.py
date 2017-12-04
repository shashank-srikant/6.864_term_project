import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.autograd as autograd
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import pandas as pd


def model_cnn(is_training_phase, num_epochs, data_to_load, idx_df, batch_size, number_negative_examples, model, criterion, optimizer, scheduler, model_name, use_gpu, save_model_at):
    print "Model invoked"
    loss_per_epoch = []
    learning_rate_schedule = []
    patience_cnt = 0

    data_loader = torch.utils.data.DataLoader(
        data_to_load, 
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4, 
        drop_last = True)

    if not is_training_phase:
        num_epochs = 1

    for epoch in range(num_epochs):
        print "epoch value: " + str(epoch)
        loss_over_batches = 0.0
        
        if is_training_phase:
            model.train()
            scheduler.step()
        else:
            model.eval()

        for batch in tqdm(data_loader):
            query_idx = batch['query_idx']
            query_title = Variable(batch['query_title'])
            query_body = Variable(batch['query_body'])
            similar_title = Variable(batch['similar_title'])
            similar_body = Variable(batch['similar_body'])

            random_title_list = []
            random_body_list = []
            for ridx in range(number_negative_examples): #number of random negative examples
                random_title_name = 'random_title_' + str(ridx)
                random_body_name = 'random_body_' + str(ridx)
                random_title_list.append(Variable(batch[random_title_name]))
                random_body_list.append(Variable(batch[random_body_name]))

            if use_gpu:
                query_title, query_body = query_title.cuda(), query_body.cuda()
                similar_title, similar_body = similar_title.cuda(), similar_body.cuda()
                random_title_list = map(lambda item: item.cuda(), random_title_list)
                random_body_list = map(lambda item: item.cuda(), random_body_list)

            if is_training_phase:
                optimizer.zero_grad()

            nn_query_title = model(query_title)
            nn_query_body = model(query_body)
            nn_query = (nn_query_title + nn_query_body)/2.0

            nn_similar_title = model(similar_title)
            nn_similar_body = model(similar_body)
            nn_similar = (nn_similar_title + nn_similar_body)/2.0

            nn_random_list = []
            for ridx in range(len(random_title_list)):
                nn_random_title = model(random_title_list[ridx])
                nn_random_body = model(random_body_list[ridx])
                nn_random = (nn_random_title + nn_random_body)/2.0
                nn_random_list.append(nn_random)
            #end for

            cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
            score_pos = cosine_similarity(nn_query, nn_similar)

            score_list = []
            score_list.append(score_pos)
            for ridx in range(len(nn_random_list)):
                score_neg = cosine_similarity(nn_query, nn_random_list[ridx])
                score_list.append(score_neg)

            X_scores = torch.stack(score_list, 1) #[batch_size, K=101]
            y_targets = Variable(torch.zeros(X_scores.size(0)).type(torch.LongTensor)) #[batch_size]
            if use_gpu:
                y_targets = y_targets.cuda()
            loss = criterion(X_scores, y_targets) #y_target=0
            
            if is_training_phase:
                loss.backward()
                optimizer.step()
            
            loss_over_batches += loss.cpu().data[0]
            
            #save scores to data-frame
            nn_query_idx = query_idx.cpu().numpy()
            nn_retrieved_scores = X_scores.cpu().data.numpy()[:,1:] #skip positive score
            for row, qidx in enumerate(nn_query_idx):
                idx_df.loc[idx_df['query_id'] == qidx, model_name] = " ".join(nn_retrieved_scores[row,:].astype('str'))
    
        #end for-batch
        loss_per_epoch.append(loss_over_batches)
    
        if is_training_phase:
            early_stop = False
            learning_rate_schedule.append(scheduler.get_lr())
            print "epoch: %4d, training loss: %.4f" %(epoch+1, loss_over_batches)
        
            torch.save(model, save_model_at)

            #early stopping
            patience = 4
            min_delta = 0.1
            if epoch > 0 and (loss_per_epoch[epoch-1] - loss_per_epoch[epoch] > min_delta):
                patience_cnt = 0
            else:
                patience_cnt += 1

            if patience_cnt > patience:
                print "early stopping..."
                early_stop = True
        
            if early_stop:
                if is_training_phase:
                    return loss_per_epoch, idx_df, learning_rate_schedule
                else:
                    return loss_per_epoch, idx_df, []
  
    #end for-epoch
    if is_training_phase:
        return loss_per_epoch, idx_df, learning_rate_schedule
    else:
        return loss_per_epoch, idx_df, []
