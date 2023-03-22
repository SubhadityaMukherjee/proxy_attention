import torch
from torch.utils.data import DataLoader, TensorDataset

def process_batches(tensors_list, batch_size, tx = None):
    # dataset = TensorDataset(torch.Tensor(tensors_list), torch.Tensor([1 for _ in range(len(tensors_list))]))
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # for batch in dataloader:
    #     print(batch)
    #     print(batch.size())
    for i in range(0, len(tensors_list), batch_size):
        print(len(tensors_list[i:i+batch_size]))
        
test_tensor = [torch.rand(10, 3, 224, 224) for x in range(10)]

process_batches(test_tensor, 3)