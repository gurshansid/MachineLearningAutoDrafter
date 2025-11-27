"""
TE Player Value Network

Evaluates TE players based on TE-specific features
"""

import torch
import torch.nn as nn


class TEPlayerNetwork(nn.Module):
    """Evaluates TE players"""
    def __init__(self, player_feature_size: int, context_size: int = 10):
        super().__init__()
        
        # Encodes TE-specific features
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Encodes draft context
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        
        # Combines both to produce TE value score
        self.decision_head = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, player_features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        player_features: [batch, player_feature_size] - TE-specific features
        context: [batch, context_size]
        returns: [batch, 1] value score for each TE
        """
        player_enc = self.player_encoder(player_features)
        context_enc = self.context_encoder(context)
        combined = torch.cat([player_enc, context_enc], dim=1)
        return self.decision_head(combined)


def get_te_features(player_row, env):
    """
    Extract TE-specific features from a player row
    
    Customize this function to select which features the TE network sees
    """
    feature_cols = [
        'fantasy_adp',
        'avg_points_per_game',
        'avg_receptions',
        'avg_receiving_yards',
        'avg_receiving_tds',
    ]
    
    features = []
    for col in feature_cols:
        norm_col = f'{col}_norm'
        if norm_col in player_row.index:
            features.append(float(player_row[norm_col]))
        else:
            features.append(0.0)
    
    return features