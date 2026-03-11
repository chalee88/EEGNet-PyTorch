from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn

def train(model, X_train, y_train, X_test, y_test, epochs=350):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    best_acc = 0.0
    patience = 50
    epochs_no_improve = 0

    history = {
        'train_acc': [],
        'test_acc': [],
        'loss': [],
        'epochs': []
    }

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            model.apply_weight_constraints()

            epoch_loss += loss.item()

            # Track training accuracy
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = epoch_loss / len(train_loader)
            train_acc = train_correct / train_total

            # Validation
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for X_test, y_test in test_loader:
                    test_outputs = model(X_test)
                    test_preds = torch.argmax(test_outputs, dim=1)
                    test_correct += (test_preds == y_test).sum().item()
                    test_total += y_test.size(0)

            test_acc = test_correct / test_total

            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['loss'].append(avg_loss)
            history['epochs'].append(epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 10

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

            print(f'Epoch {epoch} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
    return best_acc, history