"""
RB Player Value Network

Evaluates RB players based on RB-specific features
"""

import torch
import torch.nn as nn


class RBPlayerNetwork(nn.Module):
    """Evaluates RB players"""
    def __init__(self, player_feature_size: int, context_size: int = 10):
        super().__init__()
        
        # Encodes RB-specific features
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
        
        # Combines both to produce RB value score
        self.decision_head = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, player_features: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        player_features: [batch, player_feature_size] - RB-specific features
        context: [batch, context_size]
        returns: [batch, 1] value score for each RB
        """
        player_enc = self.player_encoder(player_features)
        context_enc = self.context_encoder(context)
        combined = torch.cat([player_enc, context_enc], dim=1)
        return self.decision_head(combined)


def get_rb_features(player_row, env):
    """
    Extract RB-specific features from a player row
    
    Combines rushing + receiving stats for total production
    """
    feature_cols = [
        'fantasy_adp',
        'avg_points_per_game',
        'avg_receptions',  # PPR crucial
    ]
    
    features = []
    for col in feature_cols:
        norm_col = f'{col}_norm'
        if norm_col in player_row.index:
            features.append(float(player_row[norm_col]))
        else:
            features.append(0.0)
    
    # COMBINE RUSHING + RECEIVING YARDS
    total_yards = 0.0
    if 'avg_rushing_yards' in player_row.index:
        total_yards += float(player_row['avg_rushing_yards'])
    if 'avg_receiving_yards' in player_row.index:
        total_yards += float(player_row['avg_receiving_yards'])
    
    # Normalize total yards (typical range 0-2000)
    total_yards_norm = total_yards / 1500.0
    features.append(total_yards_norm)
    
    # COMBINE RUSHING + RECEIVING TDS
    total_tds = 0.0
    if 'avg_rushing_tds' in player_row.index:
        total_tds += float(player_row['avg_rushing_tds'])
    if 'avg_receiving_tds' in player_row.index:
        total_tds += float(player_row['avg_receiving_tds'])
    
    # Normalize total TDs (typical range 0-20)
    total_tds_norm = total_tds / 20.0
    features.append(total_tds_norm)
    
    return features