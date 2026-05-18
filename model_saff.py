import torch
import torch.nn as nn
import torchvision.models as models


class SAFFOnlyModel(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.4):
        super().__init__()

        # --------------------------------
        # Visual backbone
        # --------------------------------
        backbone = models.efficientnet_b0(weights=None)

        self.visual_backbone = backbone.features
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_fc = nn.Linear(1280, feature_dim)

        # --------------------------------
        # Audio encoder
        # --------------------------------
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.audio_fc = nn.Linear(64, feature_dim)

        # --------------------------------
        # Classifier
        # --------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

        self.feature_dim = feature_dim

    def forward(self, frames, mel):
        """
        frames: (B,T,C,H,W)
        mel: (B,1,H,W)
        """

        B, T, C, H, W = frames.shape

        # =================================
        # VISUAL BRANCH
        # =================================
        frames = frames.view(B * T, C, H, W)

        v = self.visual_backbone(frames)
        v = self.visual_pool(v)

        v = v.view(B * T, -1)
        v = self.visual_fc(v)

        v = v.view(B, T, self.feature_dim)

        # =================================
        # AUDIO BRANCH
        # =================================
        a = self.audio_encoder(mel)

        a = a.mean(dim=2)
        a = a.transpose(1, 2)

        if a.shape[1] >= T:
            idx = torch.linspace(
                0,
                a.shape[1] - 1,
                T,
                device=a.device
            ).long()

            a = a[:, idx, :]
        else:
            pad = T - a.shape[1]

            a = torch.cat(
                [a, a[:, -1:, :].repeat(1, pad, 1)],
                dim=1
            )

        a = self.audio_fc(a)

        # =================================
        # SYNC SCORE
        # =================================
        sync_score = nn.functional.cosine_similarity(
            v,
            a,
            dim=-1
        )

        sync_score = sync_score.mean(dim=1)

        # =================================
        # POOL FEATURES
        # =================================
        v_pool = v.mean(dim=1)
        a_pool = a.mean(dim=1)

        fused = torch.cat(
            [
                v_pool,
                a_pool,
                sync_score.unsqueeze(1)
            ],
            dim=1
        )

        logits = self.classifier(fused)

        return logits.squeeze(1)