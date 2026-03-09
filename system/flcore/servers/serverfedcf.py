import copy
import random
import time
import torch
from system.flcore.clients.clientfedcf import clientFedCF 
from system.flcore.servers.serverbase import Server

class FedCF(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientFedCF)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.server_learning_rate = args.server_learning_rate
        
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'round time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        self.save_results()
        self.save_global_model()

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model, self.global_c)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
        
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """
        x^{r+1} = sum( q_i * x_i^{r,K} )
        g^{r+1} = sum( q_i * g_i^{r+1} )
        """
        for param in self.global_model.parameters():
            param.data.zero_()
        for c in self.global_c:
            c.data.zero_()

        for i, cid in enumerate(self.uploaded_ids):
            w = self.uploaded_weights[i]
            client = self.clients[cid]
            
            for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                server_param.data += client_param.data.clone() * w
            
            for server_c, client_c in zip(self.global_c, client.client_c):
                server_c.data += client_c.data.clone() * w

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model, self.global_c)