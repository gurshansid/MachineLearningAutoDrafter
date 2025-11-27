"""
WR Player Value Network

Evaluates WR players based on WR-specific features
"""

import torch
import torch.nn as nn


class WRPlayerNetwork(nn.Module):
    """Evaluates WR players"""
    def __init__(self, player_feature_size: int, context_size: int = 10):
        super().__init__()
        
        # Encodes WR-specific features
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
        
        # Combines both to produce WR value score
        self.decision_head = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, player_features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        player_features: [batch, player_feature_size] - WR-specific features
        context: [batch, context_size]
        returns: [batch, 1] value score for each WR
        """
        player_enc = self.player_encoder(player_features)
        context_enc = self.context_encoder(context)
        combined = torch.cat([player_enc, context_enc], dim=1)
        return self.decision_head(combined)


def get_wr_features(player_row, env):
    """
    Extract WR-specific features from a player row
    
    Customize this function to select which features the WR network sees
    """
    feature_cols = [
        'fantasy_adp',
        'avg_points_per_game',
        'avg_receptions',  # Target share is crucial for WRs
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