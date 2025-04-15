import torch

from torch.nn.utils.rnn import *


class BiLSTMPModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layer_size,
        output_size=2,
        dropout=0
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_layer_size,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.linear = torch.nn.Linear(
            2 * hidden_layer_size,  # bidirectional
            output_size
        )

    def forward(self, input_seq, lengths):
        packed_input = pack_padded_sequence(
            input_seq, lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.linear(output)
        return output


def lstm_trainer(
    dataloader,
    num_epoch,
    input_size=4,
    output_size=2,
    hidden_layer_size=32,
    learning_rate=1e-3
):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')

    model = BiLSTMPModel(input_size, hidden_layer_size, output_size).to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epoch):
        total_loss = 0.0
        total_acc = 0.0

        model.train()

        for i, batch in enumerate(dataloader):
            sample = batch['sample'].to(device)
            label = batch['label'].to(device)
            length = batch['length']

            if sample.shape[1] == 0:
                print("Empty batch detected, skipping this batch...")
                continue

            optimizer.zero_grad()
            y_pred = model(sample, length)

            y_pred = y_pred.view(-1, 2)
            label = label.view(-1)

            loss = loss_func(y_pred, label)
            loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(y_pred, dim=1)
            acc = (pred_labels == label).float()
            acc = acc.sum() / len(acc)

            total_acc += acc.item()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1} / {num_epoch} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")

    return model
            
