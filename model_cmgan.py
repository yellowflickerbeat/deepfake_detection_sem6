import torch
import torch.nn as nn
import torchvision.models as models


# ==========================================
# CMGAN ONLY MODEL
# ==========================================
class CMGANOnlyModel(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        num_heads=4,
        dropout=0.4
    ):
        super().__init__()

        # ----------------------------------
        # VISUAL ENCODER
        # ----------------------------------
        backbone = models.efficientnet_b0(weights=None)

        self.visual_backbone = backbone.features
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_fc = nn.Linear(1280, feature_dim)

        # ----------------------------------
        # AUDIO ENCODER
        # ----------------------------------
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

        # ----------------------------------
        # TRANSFORMER ENCODER
        # ----------------------------------
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=1
        )

        # ----------------------------------
        # CROSS ATTENTION
        # ----------------------------------
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # ----------------------------------
        # GATING
        # ----------------------------------
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.Sigmoid()
        )

        # ----------------------------------
        # CLASSIFIER
        # ----------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
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

        # ==================================
        # VISUAL FEATURES
        # ==================================
        frames = frames.view(B * T, C, H, W)

        v = self.visual_backbone(frames)
        v = self.visual_pool(v)

        v = v.view(B * T, -1)
        v = self.visual_fc(v)

        v = v.view(B, T, self.feature_dim)

        # ==================================
        # AUDIO FEATURES
        # ==================================
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

        # ==================================
        # TRANSFORMER ENCODING
        # ==================================
        combined = torch.cat([v, a], dim=1)

        encoded = self.transformer(combined)

        v_encoded = encoded[:, :T, :]
        a_encoded = encoded[:, T:, :]

        # ==================================
        # CROSS ATTENTION
        # ==================================
        attn_out, _ = self.cross_attention(
            query=v_encoded,
            key=a_encoded,
            value=a_encoded
        )

        # ==================================
        # FEATURE POOLING
        # ==================================
        v_pool = v_encoded.mean(dim=1)
        a_pool = a_encoded.mean(dim=1)
        attn_pool = attn_out.mean(dim=1)

        # ==================================
        # GATED FUSION
        # ==================================
        fusion_input = torch.cat(
            [v_pool, a_pool, attn_pool],
            dim=1
        )

        gate = self.gate(fusion_input)

        fused = gate * v_pool + (1 - gate) * a_pool

        # ==================================
        # CLASSIFICATION
        # ==================================
        logits = self.classifier(fused)

        return logits.squeeze(1)