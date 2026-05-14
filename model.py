import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==========================================
# SAFF MODULE
# ==========================================
class SAFFModule(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()

        # -----------------------------
        # Visual backbone
        # -----------------------------
        backbone = models.efficientnet_b0(weights=None)

        self.visual_backbone = backbone.features
        self.visual_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.visual_fc = nn.Linear(1280, feature_dim)

        # -----------------------------
        # Audio encoder (temporal)
        # -----------------------------
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

        self.feature_dim = feature_dim

    def forward(self, frames, mel):
        """
        frames: (B,T,C,H,W)
        mel: (B,1,H,W)
        """

        B, T, C, H, W = frames.shape

        # -----------------------------
        # Visual branch
        # -----------------------------
        frames = frames.view(B * T, C, H, W)

        v = self.visual_backbone(frames)
        v = self.visual_pool(v)

        v = v.view(B * T, -1)
        v = self.visual_fc(v)

        v = v.view(B, T, self.feature_dim)

        # -----------------------------
        # Audio branch
        # -----------------------------
        a = self.audio_encoder(mel)          # (B,64,H,W)

        a = a.mean(dim=2)                    # preserve temporal width
        a = a.transpose(1, 2)               # (B,time,64)

        # match video frame count
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

        # -----------------------------
        # Sync score
        # -----------------------------
        sync_score = F.cosine_similarity(
            v,
            a,
            dim=-1
        )

        sync_score = sync_score.mean(dim=1)

        return v, a, sync_score


# ==========================================
# CM-GAN MODULE
# ==========================================
class CMGANModule(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        num_heads=4,
        dropout=0.4
    ):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=1
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(self, v_feats, a_feats, sync_score):
        """
        v_feats: (B,T,D)
        a_feats: (B,T,D)
        sync_score: (B,)
        """

        combined = torch.cat(
            [v_feats, a_feats],
            dim=1
        )

        encoded = self.transformer(combined)

        T = v_feats.shape[1]

        v_encoded = encoded[:, :T, :]
        a_encoded = encoded[:, T:, :]

        attn_out, _ = self.cross_attention(
            query=v_encoded,
            key=a_encoded,
            value=a_encoded
        )

        v_pool = v_encoded.mean(dim=1)
        a_pool = a_encoded.mean(dim=1)
        attn_pool = attn_out.mean(dim=1)

        fusion_input = torch.cat(
            [v_pool, a_pool, attn_pool],
            dim=1
        )

        gate = self.gate(fusion_input)

        fused = gate * v_pool + (1 - gate) * a_pool

        sync_weight = sync_score.unsqueeze(1)
        fused = fused + 0.1 * sync_weight

        logits = self.classifier(fused)

        return logits.squeeze(1)


# ==========================================
# FULL MODEL
# ==========================================
class FullModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.saff = SAFFModule(feature_dim=256)
        self.cmgan = CMGANModule(feature_dim=256)

    def forward(self, frames, mel):
        v_feats, a_feats, sync_score = self.saff(
            frames,
            mel
        )

        logits = self.cmgan(
            v_feats,
            a_feats,
            sync_score
        )

        return logits