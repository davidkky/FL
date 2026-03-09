import torch
import numpy as np
from system.flcore.clients.clientbase import Client

class clientFedCF(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.v = args.v if hasattr(args, 'v') else 0.1 
        
        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
            
        self.global_c = None

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        initial_model = copy.deepcopy(list(self.model.parameters()))

        max_local_epochs = self.local_epochs
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()

                for param, c_i, c_g in zip(self.model.parameters(), self.client_c, self.global_c):
                    if param.grad is None:
                        continue
                    
                    denoised_grad = self.apply_dp_filter(param.grad, self.v)
                    
                    aligned_direction = denoised_grad - c_i + c_g
                    param.data -= self.learning_rate * aligned_direction

        new_client_c = []
        for p_init, p_last, c_i, c_g in zip(initial_model, self.model.parameters(), self.client_c, self.global_c):
            realized_descent = (p_init.data - p_last.data) / (max_local_epochs * len(trainloader) * self.learning_rate)
            c_i_next = c_i - c_g + realized_descent
            new_client_c.append(c_i_next)
        
        self.client_c = new_client_c

    def apply_dp_filter(self, grad, v):
        if v <= 0:
            return grad
        
        shape = grad.shape
        flat_grad = grad.view(-1)
        n = flat_grad.numel()
        
        diag = torch.ones(n, device=grad.device) * (1 + 2 * v)
        off_diag = torch.ones(n - 1, device=grad.device) * (-v)
        
        return grad / (1 + v)

    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
        self.global_c = global_c 