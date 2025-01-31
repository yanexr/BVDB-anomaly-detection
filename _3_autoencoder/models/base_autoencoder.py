import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class BaseAutoencoder(nn.Module):
    """
    config = {
        'lr': float,
        'epochs': int,
        'contains_context': bool,
        'contains_context_ecg': bool,
        'num_windows': int,
        'crop_size': int,
        'crop_begin': bool,
        'max_noise': float,
        'add_noise_channels': list[int],
        'noise_increases_to_end': bool,
        'print': bool
    }
    """
    def __init__(self, config):
        super(BaseAutoencoder, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build()
        self.to(self.device)
        
    def build(self):
        raise NotImplementedError
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self, train_loader, val_loader=None, verbose=True):
        train_losses = []
        val_losses_normal = []
        val_losses_anomaly = []
        val_auc_scores = []
        evaluation = {}

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])

        for epoch in range(self.config['epochs']):
            self.train()
            running_train_loss = 0

            for signals, _ in train_loader:
                if(self.config.get('contains_context', False)):
                    signals, _, _ = self.permutate_channels(signals)

                if(self.config.get('contains_context_ecg', False)):
                    signals = self.permutate_context_channels(signals)

                if(self.config.get('num_windows', False)):
                    signals, signals_without_remainder = self.windowing(signals)

                signals = self.crop(signals, crop_size=self.config['crop_size'])
                noisy_signals = self.add_noise(signals)
                noisy_signals = noisy_signals.to(self.device)
                if self.config.get('print', False):
                    self.config['print'] = False
                    self.print_channels(noisy_signals)
                signals = signals.to(self.device)

                optimizer.zero_grad()
                outputs = self(noisy_signals)

                if (self.config.get('num_windows', False)):
                    signals = signals_without_remainder
                    signals = signals.to(self.device)
                    outputs = self.revert_windowing(outputs)

                if(self.config.get('contains_context_ecg', False)):
                    signals = signals[:, :, 0:1]

                loss = criterion(outputs, signals)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()

            train_losses.append(running_train_loss / len(train_loader))

            if val_loader:
                evaluation = self.evaluate_model(val_loader)

                val_losses_normal.append(evaluation['avg_val_loss_normal'])
                val_losses_anomaly.append(evaluation['avg_val_loss_anomaly'])
                val_auc_scores.append(roc_auc_score(evaluation['y_true'], evaluation['y_scores']))

                if verbose:
                    print(f"Epoch {epoch + 1}/{self.config['epochs']} - Train loss: {train_losses[-1]:.4f} - Val loss normal: {val_losses_normal[-1]:.4f} - Val loss anomaly: {val_losses_anomaly[-1]:.4f} - Val AUC: {val_auc_scores[-1]:.4f}")

        return train_losses, val_losses_normal, val_losses_anomaly, val_auc_scores, evaluation
    

    def evaluate_model(self, val_loader):
        self.eval()
        criterion = nn.MSELoss()
        running_val_loss_normal = 0
        running_val_loss_anomaly = 0
        y_true = []
        y_scores = []

        with torch.no_grad():
            for signals, labels in val_loader:
                if(self.config.get('contains_context', False)):
                    signals, reversed_permutation, shape = self.permutate_channels(signals)

                if(self.config.get('contains_context_ecg', False)):
                    signals = self.permutate_context_channels(signals)

                if(self.config.get('num_windows', False)):
                    signals, signals_without_remainder = self.windowing(signals)

                if self.config.get('crop_begin', False):
                    begin = self.config['crop_size']
                    end = 0
                else:    
                    begin = self.config['crop_size'] // 2
                    end = self.config['crop_size'] - begin
                signals = self.crop(signals, begin=begin, end=end)
                noisy_signals = self.add_noise(signals)
                noisy_signals = noisy_signals.to(self.device)
                signals = signals.to(self.device)

                outputs = self(signals)

                if (self.config.get('num_windows', False)):
                    signals = signals_without_remainder
                    signals = signals.to(self.device)
                    outputs = self.revert_windowing(outputs)

                if (self.config.get('contains_context', False)):
                    signals = signals.view(shape[0], signals.shape[1], shape[2], shape[3])
                    signals = signals[:, :, :, reversed_permutation]
                    signals = signals[:, :, :, 0]
                    outputs = outputs.view(shape[0], signals.shape[1], shape[2], shape[3])
                    outputs = outputs[:, :, :, reversed_permutation]
                    outputs = outputs[:, :, :, 0]

                if(self.config.get('contains_context_ecg', False)):
                    signals = signals[:, :, 0:1]

                signals_0 = signals[labels == 0]
                outputs_0 = outputs[labels == 0]
                signals_1 = signals[labels == 1]
                outputs_1 = outputs[labels == 1]

                if len(signals_0) > 0:
                    loss_normal = criterion(outputs_0, signals_0)
                    running_val_loss_normal += loss_normal.item()

                if len(signals_1) > 0:
                    loss_anomaly = criterion(outputs_1, signals_1)
                    running_val_loss_anomaly += loss_anomaly.item()

                y_true.extend(labels.cpu().numpy())
                reconstruction_errors = torch.mean((outputs - signals) ** 2, dim=(1, 2))
                y_scores.extend(reconstruction_errors.cpu().numpy())

        return {
            'avg_val_loss_normal': running_val_loss_normal / len(val_loader),
            'avg_val_loss_anomaly': running_val_loss_anomaly / len(val_loader),
            'std_reconstruction_errors': np.std(y_scores),
            'auc_score': roc_auc_score(y_true, y_scores),
            'y_true': y_true,
            'y_scores': y_scores
        }



    def permutate_channels(self, x):
        # Permute all channels randomly
        x = x.permute(0, 1, 3, 2)
        batch_size, seq_len, n_feats, n_channels = x.shape
        permutation = torch.randperm(n_channels)  # random permutation
        x = x[:, :, :, permutation]
        reverse_permutation = torch.argsort(permutation)
        x = x.view(batch_size, seq_len, -1)
        return x, reverse_permutation, (batch_size, seq_len, n_feats, n_channels)

    def permutate_context_channels(self, x):
        # Permute all channels except the first one
        x = x.permute(0, 1, 3, 2)
        batch_size, seq_len, n_feats, n_channels = x.shape
        permutation = torch.randperm(n_channels-1)  # random permutation
        permutation = torch.cat((torch.tensor([0]), permutation+1))
        x = x[:, :, :, permutation]
        x = x.view(batch_size, seq_len, -1)
        return x



    def windowing(self, signals):
        batch_size, seq_len, channels = signals.size()
        remainder = seq_len % self.config['num_windows']

        # slice off the remainder from the beginning
        signals_without_remainder = signals[:, remainder:, :]
        seq_len = seq_len - remainder

        signals = signals_without_remainder.reshape(
            batch_size, 
            self.config['num_windows'], 
            seq_len // self.config['num_windows'], 
            channels
        )
        signals = signals.permute(0, 2, 1, 3).reshape(
            batch_size, 
            seq_len // self.config['num_windows'], 
            channels * self.config['num_windows']
        ) # (batch_size, seq_len_per_window, num_windows*channels)

        return signals, signals_without_remainder

    def revert_windowing(self, signals):
        batch_size, seq_len_per_window, merged_channels = signals.size()
        channels = merged_channels // self.config['num_windows']
        
        # Reshape back: separate windows and channels
        signals = signals.reshape(
            batch_size,
            seq_len_per_window,
            self.config['num_windows'],
            channels
        )
        
        # Permute back and merge windows into sequence length
        signals = signals.permute(0, 2, 1, 3).reshape(
            batch_size,
            seq_len_per_window * self.config['num_windows'],
            channels
        )

        return signals


    def add_noise(self, x):
        """
        Add random Gaussian noise with time-varying standard deviation to the input time series.
        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_size, channels, seq_len).
        """
        batch_size, seq_len, channels = x.shape
        x_noisy = x.clone()
        max_noise = self.config.get('max_noise', 0.2)

        if 'add_noise_channels' in self.config:
            for idx in self.config['add_noise_channels']:
                if self.config.get('noise_increases_to_end', False):
                    time = torch.linspace(-6, 6, seq_len).unsqueeze(0).to(x.device)  # (1, seq_len)
                    sigmoid_curve = torch.sigmoid(time)  # Values between 0 and 1
                    noise_stds = sigmoid_curve * max_noise  # Scale by max_noise
                    noise_stds = noise_stds.expand(batch_size, -1)  # (batch_size, seq_len)
                else:
                    noise_start = torch.rand(batch_size, 1) * max_noise
                    noise_end = torch.rand(batch_size, 1) * max_noise
                    time = torch.linspace(0, 1, seq_len).unsqueeze(0).to(x.device)  # (1, seq_len)
                    noise_stds = noise_start + (noise_end - noise_start) * time  # (batch_size, seq_len)
                
                # Generate noise with time-varying std
                noise = torch.normal(mean=0, std=noise_stds)  # (batch_size, seq_len)
                # add noise to the given channel
                x_noisy[:, :, idx] += noise
                           
        return x_noisy


    def print_channels(self, x):
        batch_size, seq_len, channels = x.shape
        for i in range(channels):
            plt.plot(x[0, :, i].cpu().numpy(), label=f'Channel {i}')
        plt.show()
    


    def crop(self, x, begin=None, end=None, crop_size=None):
        """
        Crop the time series from the beginning and end.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch_size, channels, seq_len).
        begin : int, optional
            Number of points to crop from the beginning.
        end : int, optional
            Number of points to crop from the end.
        amount : int, optional
            If begin and end are None, random begin and end values are generated for each batch element, where begin + end = amount.
        
        """
        if crop_size == 0:
            return x

        if begin is None and end is None:
            x_cropped = torch.zeros(x.size(0), x.size(1)-crop_size, x.size(2))
            for i in range(x.size(0)):
                begin = torch.randint(0, crop_size, (1,))
                end = crop_size - begin
                if end == 0:
                    x_cropped[i] = x[i, begin: , : ]
                else:
                    x_cropped[i] = x[i, begin:-end, :]
        else:
            if end == 0 or end is None:
                x_cropped = x[:, begin: , :]
            elif begin == 0 or begin is None:
                x_cropped = x[:, :-end, :]
            else:
                x_cropped = x[:, begin:-end, :]

        return x_cropped



    def plot_reconstruction(self, x, figsize=(10, 5), index=0):
        self.eval()
        with torch.no_grad():
            if(self.config.get('contains_context', False)):
                    x, reversed_permutation, shape = self.permutate_channels(x)

            if(self.config.get('contains_context_ecg', False)):
                    x = self.permutate_context_channels(x)
            
            if(self.config.get('num_windows', False)):
                    x, x_without_remainder = self.windowing(x)

            begin = self.config['crop_size'] // 2
            end = self.config['crop_size'] - begin
            x = self.crop(x, begin=begin, end=end)
            noisy_x = self.add_noise(x)
            noisy_x = noisy_x.to(self.device)
            x = x.to(self.device)

            reconstructed = self(noisy_x)

            if (self.config.get('num_windows', False)):
                    x = x_without_remainder
                    x = x.to(self.device)
                    reconstructed = self.revert_windowing(reconstructed)

            if (self.config.get('contains_context', False)):
                    reconstructed = reconstructed.view(shape[0], reconstructed.shape[1], shape[2], shape[3])
                    reconstructed = reconstructed[:, :, :, reversed_permutation]
                    reconstructed = reconstructed[:, :, :, 0]
                    x = x.view(shape[0], reconstructed.shape[1], shape[2], shape[3])
                    x = x[:, :, :, reversed_permutation]
                    x = x[:, :, :, 0]

            if(self.config.get('contains_context_ecg', False)):
                    x = x[:, :, 0:1]

            reconstruction_error = torch.mean((x - reconstructed)**2, dim=(1,2))
            print(f'Reconstruction error: {reconstruction_error.item()}')

            x = x.cpu()
            reconstructed = reconstructed.cpu()

            rec = reconstructed[0,:,index]
            ori = x[0,:,index]

            plt.figure(figsize=figsize)
            plt.plot(rec, label='Reconstructed', color='tab:red')
            plt.plot(ori, label='Original', color='black')
            plt.fill_between(range(len(rec)), rec, ori, color='tab:red', alpha=0.2, label='Error')
            plt.legend()
            plt.show()


    def print_num_parameters(self):
        num_parameters = sum(p.numel() for p in self.parameters())
        print(f'Number of parameters: {num_parameters}')
        num_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Number of trainable parameters: {num_trainable_parameters}')
    

    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    