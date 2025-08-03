import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes, model_name="facebook/wav2vec2-base", freeze_encoder=True, num_speakers=None, speaker_emb_dim=32):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.encoder.encoder.layers[-2:].parameters():
                param.requires_grad = True
        self.use_speaker = num_speakers is not None and num_speakers > 0
        if self.use_speaker:
            self.speaker_emb = nn.Embedding(num_speakers, speaker_emb_dim)
            classifier_in_dim = self.encoder.config.hidden_size + speaker_emb_dim
        else:
            classifier_in_dim = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    def forward(self, input_values, speaker_id=None):
        try:
            outputs = self.encoder(input_values)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)
            if self.use_speaker and speaker_id is not None:
                spk_emb = self.speaker_emb(speaker_id)
                pooled = torch.cat([pooled, spk_emb], dim=1)
            logits = self.classifier(pooled)
            return logits
        except Exception as e:
            print(f"[ERROR] Model forward pass failed: {e}")
            raise
