import torch
import torch.nn as nn

'''
The Input Features Shape Will Be:
    channel = {static, delta, delta-delta}
    [Batch, channel, n_frames, n_mels]
'''

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r), nn.ReLU(),
            nn.Linear(ch//r, ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.se(x).view(x.shape[0], -1, 1, 1)

class CNNClassifier(nn.Module):
    def __init__(self, n_classes: int, 
                 average_frames: bool = False,use_deltas=True,
                 *args, **kwargs):
        super().__init__()
        self.average_frames = average_frames
        self.use_deltas = use_deltas
        k_channels = 3 if use_deltas else 1

        self.conv = nn.Sequential(
            nn.Conv2d(k_channels, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2,2)), nn.Dropout2d(0.25),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), 
            nn.Linear(512, 256), nn.ReLU(), 
            nn.Dropout(0.50),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        if self.average_frames:
            x = x.mean(dim=2,keepdim=True) 
        if not self.use_deltas:
            x = x[:,0:1,...]
        return self.head(self.pool(self.conv(x)))

class CNNSEClassifier(nn.Module):
    def __init__(self, n_classes: int, 
                 average_frames: bool = False,use_deltas=True,
                 *args, **kwargs):
        super().__init__()
        self.average_frames = average_frames
        self.use_deltas = use_deltas
        k_channels = 3 if use_deltas else 1

        self.conv = nn.Sequential(
            nn.Conv2d(k_channels, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            nn.Dropout2d(0.25),
            SEBlock(32),

            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            nn.Dropout2d(0.25),
            SEBlock(64),

            nn.Conv2d(64, 128, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d((2,2)), 
            nn.Dropout2d(0.25),
            SEBlock(128),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), 
            nn.Linear(512, 256), nn.ReLU(), 
            nn.Dropout(0.50),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        if self.average_frames:
            x = x.mean(dim=2,keepdim=True) 
        if not self.use_deltas:
            x = x[:,0:1,...]
        return self.head(self.pool(self.conv(x)))

class DenseClassifier(nn.Module):
    def __init__(self, n_classes: int, n_frames: int, n_mels: int = 16,
                 average_frames: bool = False,use_deltas=True,*args, **kwargs):
        super().__init__()
        self.average_frames = average_frames
        self.use_deltas = use_deltas

        k_channels = 3 if use_deltas else 1
    
        if self.average_frames:
            in_features = k_channels * n_mels
        else:
            in_features = k_channels * n_frames * n_mels
            
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        if self.average_frames:
            x = x.mean(dim=2,keepdim=True) 
        if not self.use_deltas:
            x = x[:,0:1,...]
        return self.net(x)


class CRNNClassifier(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = 16, *args, **kwargs):    
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=(1, 2)), 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        
        cnn_out_freq = n_mels // 4 
        rnn_in_size = 64 * cnn_out_freq
        
        self.rnn = nn.GRU(input_size=rnn_in_size, hidden_size=128, 
                          num_layers=2, batch_first=True, dropout=0.3)
        
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x) # Output: [B, 64, T, F//4]
        
        # Prepare for RNN: PyTorch RNNs expect [Batch, Time, Features]
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3) # Change to [B, T, C, F]
        x = x.reshape(B, T, C * F) # Flatten C and F into one feature vector: [B, T, C*F]
        
        rnn_out, _ = self.rnn(x) # rnn_out: [Batch, Time, Hidden_Size]
        
        # Grab the output of the VERY LAST time step to pass to the classifier
        last_time_step = rnn_out[:, -1, :] # [Batch, Hidden_Size]
        
        return self.head(last_time_step)

class LSTMClassifier(nn.Module):
    """A pure RNN approach without CNN feature extraction."""
    def __init__(self, n_classes: int, n_mels: int = 16, *args, **kwargs):
        super().__init__()
        # Flatten channels and mels into a single feature vector per time step
        in_features = 3 * n_mels 
        
        self.lstm = nn.LSTM(input_size=in_features, hidden_size=512, 
                            num_layers=2, batch_first=True, dropout=0.5)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        B, C, T, F = x.size()
        # Rearrange to [Batch, Time, Channels, Mels] then flatten C and F
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the output from the last time step
        last_time_step = lstm_out[:, -1, :] 
        return self.head(last_time_step)


class TCNClassifier(nn.Module):
    """Temporal Convolutional Network: Uses 1D convs across the time dimension."""
    def __init__(self, n_classes: int, n_mels: int = 16, *args, **kwargs):
        super().__init__()
        in_channels = 3 * n_mels
        
        # Dilated 1D convolutions map temporal patterns rapidly
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        B, C, T, F = x.size()
        # Conv1d expects [Batch, Channels, Time]. We treat (C*F) as the channels.
        x = x.permute(0, 1, 3, 2).reshape(B, C * F, T)
        
        x = self.tcn(x)               # [B, 256, T]
        x = self.pool(x).squeeze(-1)  # [B, 256]
        return self.head(x)


class CNNTransformer(nn.Module):
    """CNN for feature extraction, Transformer for sequence modeling."""
    def __init__(self, n_classes: int, n_mels: int = 16, *args, **kwargs):
        super().__init__()
        
        # CNN frontend (downsamples frequency to make sequence lighter)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )
        
        cnn_out_freq = n_mels // 4 
        d_model = 64 * cnn_out_freq
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=d_model*2, 
            dropout=0.3, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.cnn(x) # [B, 64, T, F//4]
        
        B, C, T, F = x.size()
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F) # [B, T, d_model]
        
        # Pass through transformer
        x = self.transformer(x) # [B, T, d_model]
        
        # Global average pooling over the time dimension
        x = x.mean(dim=1) # [B, d_model]
        return self.head(x)


AVAILABLE_MODELS = {
    "CNN"         : CNNClassifier,
    "CNNSE"       : CNNSEClassifier,
    "Dense"       : DenseClassifier,
    "CRNN"        : CRNNClassifier,
    "LSTM"        : LSTMClassifier,
    "TCN"         : TCNClassifier,
    "Transformer" : CNNTransformer
}