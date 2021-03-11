

def eval_func(model,data_iter,device):
    model.eval()
    with torch.no_grad():
        count, total_count = 0,0
        for batch_in, batch_label in data_iter:
            x = batch_in.to(device)
            y = batch_label.to(device)
            y_pred = model.forward(x)
            y_ = torch.argmax(y_pred, dim=-1)
            count += (y==y_).sum().item()
            total_count += batch_in.size(0)
    model.train()
    return count/total_count