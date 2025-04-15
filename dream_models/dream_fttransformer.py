import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import rtdl
import numpy as np


def ft_transformer_trainer(
    X_train_num, X_val_num, 
    X_train_cat, X_val_cat,
    y_train, y_val, 
    num_epochs=10,
    batch_size=32,
    learning_rate=1e-3,
    cat_cardinalities=[],
    d_token=32,
    n_blocks=6,
    num_classes=2,
    attention_dropout=0.1,
    ffn_d_hidden=64,
    ffn_dropout=0.1,
    residual_dropout=0.1,
):

    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train)
    val_dataset = TensorDataset(X_val_num, X_val_cat, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_features = X_train_num.shape[1]
    
    model = rtdl.FTTransformer.make_baseline(
        n_num_features          = num_features,         # number of features
        cat_cardinalities       = cat_cardinalities,    # cardinality of categorical features
        d_token                 = d_token,              # dimensionality of tokens
        n_blocks                = n_blocks,             # number of Transformer blocks
        d_out                   = num_classes,          # number of classes
        attention_dropout       = attention_dropout,    # attention dropout ratio
        ffn_d_hidden            = ffn_d_hidden,         # dimensionality of hidden layer in feedforward network
        ffn_dropout             = ffn_dropout,          # feedforward dropout ratio
        residual_dropout        = residual_dropout,     # residual dropout ratio
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Current device: {device}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=5, verbose=True
    # )
    
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for num_inputs, cat_inputs, labels in train_loader:
            num_inputs = num_inputs.to(device)
            cat_inputs = cat_inputs.to(device)
            labels = labels.to(device)

            outputs = model(num_inputs, cat_inputs)

            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for num_inputs, cat_inputs, labels in val_loader:
                num_inputs = num_inputs.to(device)
                cat_inputs = cat_inputs.to(device)
                labels = labels.to(device)

                outputs = model(num_inputs, cat_inputs)
                probas = torch.softmax(outputs, dim=1)
                _, pred_labels = torch.max(outputs, 1)

                all_preds.extend(pred_labels.cpu().numpy())
                all_probs.extend(probas[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}, Val Accuracy: {accuracy:.4f}, Val AUC: {auc:.4f}')

        # scheduler.step(auc)
    
    return model, all_preds, all_probs, all_labels
