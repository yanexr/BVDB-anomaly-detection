import torch.nn as nn
from .base_autoencoder import BaseAutoencoder


def conv_block(in_channels, out_channels, config):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=config['conv']['ker'], stride=2, padding=config['conv']['pad'], padding_mode='replicate'),
        nn.BatchNorm1d(out_channels) if config['batch_norm'] else nn.Identity(),
        nn.ReLU() if config['activation'] == 'relu' else nn.Identity(),
        nn.Dropout(config['enc_dropout']),
    )

def deconv_block(in_channels, out_channels, config, final=False):
    return nn.Sequential(
        nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=config['conv']['ker'], stride=2, padding=config['conv']['pad'], output_padding=1),
        nn.BatchNorm1d(out_channels) if config['batch_norm'] and not final else nn.Identity(),
        nn.ReLU() if config['activation'] == 'relu' and not final else nn.Identity(),
        nn.Dropout(config['dec_dropout']) if not final else nn.Identity(),
    )


class EDA_CNN(BaseAutoencoder):
    def build(self):
        self.encoder = nn.Sequential(
            conv_block(1, 8, self.config),  # 88 → 44
            conv_block(8, 16, self.config),  # 44 → 22
            conv_block(16, 32, self.config),  # 22 → 11
            nn.Flatten(),
            nn.Linear(in_features=32 * 11, out_features=self.config['latent_dim']),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.config['latent_dim'], out_features=32 * 11),
            nn.Unflatten(1, (32, 11)),
            deconv_block(32, 16, self.config),  # 11 → 22
            deconv_block(16, 8, self.config),  # 22 → 44
            deconv_block(8, 1, self.config, final=True),  # 44 → 88
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)

        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x
    

class EMG_CNN(BaseAutoencoder):
    def build(self):
        self.encoder = nn.Sequential(
            conv_block(1, 8, self.config),  # 704 → 352
            conv_block(8, 16, self.config),  # 352 → 176
            conv_block(16, 16, self.config),  # 176 → 88
            conv_block(16, 32, self.config),  # 88 → 44
            conv_block(32, 64, self.config),  # 44 → 22
            conv_block(64, 128, self.config),  # 22 → 11
        )

        self.decoder = nn.Sequential(
            deconv_block(128, 64, self.config),  # 11 → 22
            deconv_block(64, 32, self.config),  # 22 → 44
            deconv_block(32, 16, self.config),  # 44 → 88
            deconv_block(16, 16, self.config),  # 88 → 176
            deconv_block(16, 8, self.config),  # 176 → 352
            deconv_block(8, 1, self.config, final=True),  # 352 → 704
        )

        self.linear_enc = nn.Linear(in_features=128, out_features=self.config['latent_dim'])
        self.linear_dec = nn.Linear(in_features=self.config['latent_dim'], out_features=128)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        x = x.permute(0, 2, 1)

        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x
    

class ECG_CNN(BaseAutoencoder):
    def build(self):
        self.encoder = nn.Sequential(
            conv_block(1, 16, self.config),  # 1152 → 576
            conv_block(16, 32, self.config),  # 576 → 288
            conv_block(32, 32, self.config),  # 288 → 144
            conv_block(32, 64, self.config),  # 144 → 72
            conv_block(64, 128, self.config),  # 72 → 36
            conv_block(128, 128, self.config),  # 36 → 18
        )

        self.decoder = nn.Sequential(
            deconv_block(128, 128, self.config),  # 18 → 36
            deconv_block(128, 64, self.config),  # 36 → 72
            deconv_block(64, 32, self.config),  # 72 → 144
            deconv_block(32, 32, self.config),  # 144 → 288
            deconv_block(32, 16, self.config),  # 288 → 576
            deconv_block(16, 1, self.config, final=True),  # 576 -> 1152
        )

        self.linear_enc = nn.Linear(in_features=128, out_features=self.config['latent_dim'])
        self.linear_dec = nn.Linear(in_features=self.config['latent_dim'], out_features=128)

        #self.enc_lstm = nn.LSTM(input_size=128, hidden_size=self.config['latent_dim'], num_layers=1, batch_first=True)
        #self.dec_lstm = nn.LSTM(input_size=self.config['latent_dim'], hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        x = x.permute(0, 2, 1)

        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x
    

class CNN(BaseAutoencoder):
    def build(self):
        self.encoder = nn.Sequential(
            conv_block(3, 32, self.config),  # 1152 → 576
            conv_block(32, 32, self.config),  # 576 → 288
            conv_block(32, 64, self.config),  # 288 → 144
            conv_block(64, 64, self.config),  # 144 → 72
            conv_block(64, 128, self.config),  # 72 → 36
        )

        self.decoder = nn.Sequential(
            deconv_block(128, 64, self.config),  # 36 → 72
            deconv_block(64, 64, self.config),  # 72 → 144
            deconv_block(64, 32, self.config),  # 144 → 288
            deconv_block(32, 32, self.config),  # 288 → 576
            deconv_block(32, 3, self.config, final=True),  # 576 -> 1152
        )

        self.linear_enc = nn.Linear(in_features=128, out_features=self.config['latent_dim'])
        self.linear_dec = nn.Linear(in_features=self.config['latent_dim'], out_features=128)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        x = x.permute(0, 2, 1)

        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x
    

########################
class ECG_CNNAutoencoder(BaseAutoencoder):
    def build(self):
        self.encoder = nn.Sequential(
            conv_block(1, 16),  # 1408 → 704
            conv_block(16, 32),  # 704 → 352
            conv_block(32, 32),  # 352 → 176
            conv_block(32, 64),  # 176 → 88
            conv_block(64, 128),  # 88 → 44
            conv_block(128, 128),  # 44 → 22
        )

        self.decoder = nn.Sequential(
            deconv_block(128, 128),  # 22 → 44
            deconv_block(128, 64),  # 44 → 88
            deconv_block(64, 32),  # 88 → 176
            deconv_block(32, 32),  # 176 → 352
            deconv_block(32, 16),  # 352 → 704
            deconv_block(16, 1, final=True),  # 704 → 1408
        )

        self.linear_enc = nn.Linear(in_features=128, out_features=self.config['latent_dim'])
        self.linear_dec = nn.Linear(in_features=self.config['latent_dim'], out_features=128)

        #self.enc_lstm = nn.LSTM(input_size=128, hidden_size=self.config['latent_dim'], num_layers=1, batch_first=True)
        #self.dec_lstm = nn.LSTM(input_size=self.config['latent_dim'], hidden_size=128, num_layers=1, batch_first=True)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder(x)

        x = x.permute(0, 2, 1)
        x = self.linear_enc(x)
        x = self.linear_dec(x)
        x = x.permute(0, 2, 1)

        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x