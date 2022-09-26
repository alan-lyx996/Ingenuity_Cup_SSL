from torch import nn
import torch

class TFC(nn.Module): # Frequency domain encoder
    def __init__(self, params):
        super(TFC, self).__init__()

        self.conv_block1_t = nn.Sequential(
            nn.Conv1d(params["input_channels"], 32, kernel_size=params["kernel_size"],
                      stride=params["stride"], bias=False, padding=(params["kernel_size"]//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(params["dropout"])
        )

        self.conv_block2_t = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3_t = nn.Sequential(
            nn.Conv1d(64, params["final_out_channels"], kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(params["final_out_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_t = nn.Sequential(
            nn.Linear(params["CNNoutput_channel"] * params["final_out_channels"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.conv_block1_f = nn.Sequential(
            nn.Conv1d(params["input_channels"], 32, kernel_size=params["kernel_size"],
                      stride=params["stride"], bias=False, padding=(params["kernel_size"] // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(params["dropout"])
        )

        self.conv_block2_f = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3_f = nn.Sequential(
            nn.Conv1d(64, params["final_out_channels"], kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(params["final_out_channels"]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projector_f = nn.Sequential(
            nn.Linear(params["CNNoutput_channel"] * params["final_out_channels"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):

        """Time-based Contrastive Encoder"""
        x = self.conv_block1_t(x_in_t)
        x = self.conv_block2_t(x)
        x = self.conv_block3_t(x)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.conv_block1_f(x_in_f)
        f = self.conv_block2_f(f)
        f = self.conv_block3_f(f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module): # Frequency domain encoder
    def __init__(self, params):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, params["num_classes_target"])

    def forward(self, emb):
        # """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
