import torch
import os
import wandb

def train_loop(dataloader, model, loss_fn, optimizer, device, wandb=None):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    train_loss, correct = 0, 0

    for batch, (batch_input, batch_labels) in enumerate(dataloader):
      batch_input = torch.squeeze(batch_input, 1)
      pred = model(batch_input.to(device))
      pred = pred.squeeze()
      loss = loss_fn(pred, batch_labels.to(device))
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      train_loss += loss_fn(pred, batch_labels.to(device)).item()
    #   correct += (pred.argmax(1) == batch_labels.to(device)).type(torch.float).sum().item()

      if batch % 5 == 0:
            # loss, current = loss.item(), (batch + 1) * len(batch_input)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            loss = loss.item()
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f}")

    train_loss /= num_batches
    # correct /= size

    if wandb:
        wandb.log({"train loss": train_loss})
        # wandb.log({"train accuracy": 100*correct})

    # return train_loss, 100*correct
    print(f"training Error: Avg val loss: {train_loss:>8f} \n")

    return train_loss


def val_loop(dataloader, model, loss_fn, device, wandb=None):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0


    with torch.no_grad():
        for batch_input, batch_labels in dataloader:
            batch_input = torch.squeeze(batch_input, 1)
            pred = model(batch_input.to(device))
            pred = pred.squeeze()
            val_loss += loss_fn(pred, batch_labels.to(device)).item()
            # correct += (pred.argmax(1) == batch_labels.to(device)).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size

    if wandb:
        wandb.log({"val loss": val_loss})
        # wandb.log({"val accuracy": 100*correct})

    # print(f"val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg val loss: {val_loss:>8f} \n")
    # return val_loss, 100*correct

    print(f"val Error: Avg val loss: {val_loss:>8f} \n")
    return val_loss


def train_model(args, model, train_loader, val_loader, optimizer, loss_fn, device, output_path, wandb=None, fold = None):
    model.train()
    if wandb:
        run_id = wandb.run.id
        run_name = wandb.run.name
    else:
        run_id = "local"
        run_name = "local"

    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train_loop(train_loader, model, loss_fn, optimizer, device, wandb)
        val_loss = val_loop(val_loader, model, loss_fn, device, wandb)


        torch.save({
                'epoch': t,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train loss': train_loss,
                'val loss': val_loss,
                # 'train accuracy': train_acc,
                # 'val accuracy': val_acc,
                "run Id":run_id,
                "run name":run_name,
                }, output_path)
    print("Done!")


def map_labels(outputs):
    labels = []
    for i in outputs:
        if i < 25:
            labels.append(1)
        elif i < 50:
            labels.append(2)
        elif i < 75:
            labels.append(3)
        else:
            labels.append(4)
    return labels


def test_model(args, model, test_loader, loss_fn, device, wandb=None):
    model.eval()
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    test_loss, correct = 0, 0


    with torch.no_grad():
        for batch_input, batch_labels in test_loader:
            batch_input = torch.squeeze(batch_input, 1)
            pred = model(batch_input.to(device))
            pred = pred.squeeze()
            test_loss += loss_fn(pred, batch_labels.to(device)).item()
            pred_labels = torch.Tensor(map_labels(pred))
            correct_labels = torch.Tensor(map_labels(batch_labels.to(device)))
            correct += (pred_labels.to(device) == correct_labels.to(device)).type(torch.float).sum().item()
            # correct += (pred.argmax(1) == batch_labels.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    correct /= size

    if wandb:
        wandb.log({"test loss": test_loss})
        wandb.log({"test accuracy": 100*correct})

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # return test_loss, 100*correct

    # print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss