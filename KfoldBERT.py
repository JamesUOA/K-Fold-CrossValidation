import torch
import numpy as np
import time
import datetime
import random

from Kfold import KFold
from split_data import DataManager
from transformers import BertTokenizer
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

class KfoldBERTData(DataManager):
    
    def __init__(self, data, labels, num_folds):
        super().__init__(data, labels, num_folds)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def pre_process(self, sentences, labels):
                
        max_len = 0
        for sent in sentences:
            input_ids = self.tokenizer.encode(sent, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        input_ids = []
        attention_masks = []

        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                                sent,                      
                                add_special_tokens = True, 
                                max_length = 350,          
                                pad_to_max_length = True,
                                return_attention_mask = True,
                                return_tensors = 'pt',
                                truncation=True
                        )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        dataset = TensorDataset(input_ids, attention_masks, labels)
        d, _ = random_split(dataset, [len(dataset), 0])
        return d

class KfoldBERT(KFold):
    def __init__(self, data, labels, num_folds):
        super().__init__(data, labels, num_folds)
        self.batch_size = 8
        self.epochs = 10
        self.data = KfoldBERTData(data, labels, num_folds)

        if torch.cuda.is_available():    
            self.device = torch.device("cuda") 

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, time):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((time)))
        return str(datetime.timedelta(seconds=elapsed_rounded))       

    def train(self, train_dataset, val_dataset):

        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = self.batch_size # Trains with this batch size.
                )
        validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = self.batch_size # Evaluate with this batch size.
        )

        model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                num_labels = 4, # The number of output labels--2 for binary classification.
                                # You can increase this for multi-class tasks.   
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
                )

        model.cuda()

        optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        total_steps = len(train_dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()

                
        # For each epoch...
        for epoch_i in range(0, self.epochs):
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                model.zero_grad()        
                loss, logits = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask, 
                                    labels=b_labels)
        
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            
            
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()
            model.eval()

            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            for batch in validation_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)
                
                with torch.no_grad():        
                    loss, logits = model(b_input_ids, 
                                        token_type_ids=None, 
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                # Accumulate the validation loss.
                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
                

            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
            torch.save(model.state_dict(), "removed_model_epoch_" + str(epoch_i + 1) +".pth")
        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))

        return avg_val_accuracy, avg_val_loss

