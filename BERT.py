import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, AdamW


class SentimentAnalysisDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.data.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.Y[index], dtype=torch.float)
        }

    def __len__(self):
        return self.len


# Load your data
data = pd.read_csv("testdata.csv")
slogan_data = pd.read_csv("DatasetPresidentialSlogans.csv", sep=';', on_bad_lines='skip')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the DataLoader
MAX_LEN = 64
BATCH_SIZE = 16
train_size = 0.8

train_dataset = data.sample(frac=train_size, random_state=42)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

training_set = SentimentAnalysisDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = SentimentAnalysisDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Instantiate the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to('cpu')

# Define the optimizer
optimizer = AdamW(params=model.parameters(), lr=1e-5)


# Define the training loop
def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to('cpu', dtype=torch.long)
        mask = data['mask'].to('cpu', dtype=torch.long)
        targets = data['targets'].to('cpu', dtype=torch.float)

        outputs = model(ids, mask)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.logits.squeeze()[:, 0], targets)
        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Train the model
EPOCHS = 10
# Train the BERT model
for epoch in range(EPOCHS):
    train(epoch)

# Now apply the trained model to the Slogan dataset
def sentiment_analysis(data):
    model.eval()
    sentiments = []
    for slogan in data['Slogan']:
        inputs = tokenizer.encode_plus(
            slogan,
            None,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to('cuda')
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to('cuda')
        with torch.no_grad():
            outputs = model(ids, mask)
        sentiment = torch.sigmoid(outputs.logits).item()
        sentiments.append(sentiment)

    return sentiments

# Apply the model to the slogans
slogan_data['Sentiment'] = sentiment_analysis(slogan_data)
print(slogan_data)