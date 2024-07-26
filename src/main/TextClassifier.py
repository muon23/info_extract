import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset


# Define a simple text classifier model with an embedding layer
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Sequential(
            nn.Linear(embed_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.mean(dim=1)  # Average over the sequence length
        output = self.fc(embedded)
        return output


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Define the model with pre-trained BERT and a classifier on top
class BertTextClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BertTextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # [CLS] token representation, could also use outputs['pooler_output']
        output = self.fc(pooled_output)
        return output

# Configurations
BERT_MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 2
MAX_LEN = 32
BATCH_SIZE = 2
EPOCHS = 5

# Dummy data
train_texts = ["I love machine learning", "Deep learning is amazing", "NLP is so interesting",
               "I dislike math", "Calculus is boring", "Linear algebra is tough"]
train_labels = [0, 0, 0, 1, 1, 1]
test_texts = ["I like programming", "Math is difficult"]
test_labels = [0, 1]

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Create datasets and dataloaders
train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize the model, loss function, and optimizer
model = BertTextClassifier(BERT_MODEL_NAME, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = model(input_ids, attention_mask)
        _, predicted = torch.max(outputs.data, 1)
        print(f'Texts: {batch["text"]}')
        print(f'Labels: {labels.tolist()}')
        print(f'Predicted: {predicted.tolist()}')