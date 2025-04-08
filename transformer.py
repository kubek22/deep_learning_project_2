import torch
import torch.nn as nn

class SpeechTransformer(nn.Module):
    def __init__(self, num_classes=11, input_dim=81, seq_len=201, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super(SpeechTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.squeeze(1) 
        x = self.input_projection(x) 
        x = x + self.positional_encoding 
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)

        out = self.classifier(x) 
        return out