"""
FIXED HIERARCHICAL RL - All 5 Networks in One File

Fixes:
1. Feature extraction (was returning all zeros!)
2. Proper normalization handling
3. Correct column names from your data
4. Added restart logic
5. Bot slippage for easier learning

Run this single file - no separate imports needed
"""

import pandas as pd
import numpy as np
import random
from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy


# ============================================================================
# POSITION NETWORK - Decides QB/RB/WR/TE
# ============================================================================

class PositionPolicyNetwork(nn.Module):
    """Decides which position to draft given context"""
    def __init__(self, context_size: int = 10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(context_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # QB, RB, WR, TE
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


# ============================================================================
# PLAYER NETWORKS - One per position (all same architecture for simplicity)
# ============================================================================

class PlayerValueNetwork(nn.Module):
    """Generic player value network (used for all positions)"""
    def __init__(self, player_feature_size: int, context_size: int = 10):
        super().__init__()

        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decision_head = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, player_features: torch.Tensor, context: torch.Tensor):
        player_enc = self.player_encoder(player_features)
        context_enc = self.context_encoder(context)
        combined = torch.cat([player_enc, context_enc], dim=1)
        return self.decision_head(combined)


# ============================================================================
# FEATURE EXTRACTION - FIXED VERSION
# ============================================================================

def get_qb_features(player_row, feature_means, feature_stds):
    """Extract and normalize QB-specific features"""

    # Use columns that ACTUALLY exist in your data
    feature_cols = [
        'fantasy_adp',
        'avg_points_per_game',
        'avg_passing_yards',
        'avg_passing_tds',
        'avg_rushing_yards',  # QB rushing matters
        'avg_rushing_tds'
    ]

    features = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            # Normalize using precomputed stats
            if col in feature_means.index:
                mean = feature_means[col]
                std = feature_stds[col]
                normalized = (val - mean) / std if std > 0 else 0.0
            else:
                normalized = val / 100.0  # Fallback normalization
            features.append(normalized)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def get_rb_features(player_row, feature_means, feature_stds):
    """Extract and normalize RB-specific features"""

    feature_cols = [
        'fantasy_adp',
        'avg_points_per_game',
        'avg_rushing_yards',
        'avg_rushing_tds',
        'avg_receptions',      # PPR crucial for RBs
        'avg_receiving_yards',
        'avg_receiving_tds'
    ]

    features = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            if col in feature_means.index:
                mean = feature_means[col]
                std = feature_stds[col]
                normalized = (val - mean) / std if std > 0 else 0.0
            else:
                normalized = val / 100.0
            features.append(normalized)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def get_wr_features(player_row, feature_means, feature_stds):
    """Extract and normalize WR-specific features"""

    feature_cols = [
        'fantasy_adp',
        'avg_points_per_game',
        'avg_targets',         # Target share is key
        'avg_receptions',
        'avg_receiving_yards',
        'avg_receiving_tds'
    ]

    features = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            if col in feature_means.index:
                mean = feature_means[col]
                std = feature_stds[col]
                normalized = (val - mean) / std if std > 0 else 0.0
            else:
                normalized = val / 100.0
            features.append(normalized)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


def get_te_features(player_row, feature_means, feature_stds):
    """Extract and normalize TE-specific features"""

    feature_cols = [
        'fantasy_adp',
        'avg_points_per_game',
        'avg_targets',
        'avg_receptions',
        'avg_receiving_yards',
        'avg_receiving_tds'
    ]

    features = []
    for col in feature_cols:
        if col in player_row.index:
            val = float(player_row[col])
            if col in feature_means.index:
                mean = feature_means[col]
                std = feature_stds[col]
                normalized = (val - mean) / std if std > 0 else 0.0
            else:
                normalized = val / 100.0
            features.append(normalized)
        else:
            features.append(0.0)

    return np.array(features, dtype=np.float32)


# ============================================================================
# ENVIRONMENT
# ============================================================================

class DraftEnvironment:
    def __init__(self, condensed_path: str, stats_path: str, n_teams=12, n_rounds=8, seed=42):
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        self.rng = random.Random(seed)

        self.roster_limits = {"QB": 2, "RB": 5, "WR": 5, "TE": 3}
        self.starting_requirements = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}

        # Load condensed (has features for evaluation)
        df_condensed = pd.read_csv(condensed_path)

        # Convert position one-hot to single column
        if 'pos_qb' in df_condensed.columns:
            df_condensed['position'] = 'UNKNOWN'
            df_condensed.loc[df_condensed['pos_qb'] == 1, 'position'] = 'QB'
            df_condensed.loc[df_condensed['pos_rb'] == 1, 'position'] = 'RB'
            df_condensed.loc[df_condensed['pos_wr'] == 1, 'position'] = 'WR'
            df_condensed.loc[df_condensed['pos_te'] == 1, 'position'] = 'TE'

        # Load 2024 stats (has actual fantasy points for scoring)
        df_2024 = pd.read_csv(stats_path)

        # Merge to get actual 2024 fantasy points
        df = df_condensed.merge(
            df_2024[['first_name', 'last_name', 'fantasy_points_ppr']],
            on=['first_name', 'last_name'],
            how='inner',  # Only keep players in both files
            suffixes=('_pred', '_actual')
        )

        # Use actual 2024 points for scoring
        if 'fantasy_points_ppr_actual' in df.columns:
            df['fantasy_points_ppr'] = df['fantasy_points_ppr_actual']

        # Filter to draftable
        df = df[df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
        df = df[df['fantasy_adp'] < 300].copy()
        df = df.sort_values('fantasy_adp').reset_index(drop=True)

        self.player_pool = df

        # Compute feature statistics for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.feature_means = df[numeric_cols].mean()
        self.feature_stds = df[numeric_cols].std().replace(0, 1.0)

        self.reset()
        print(f"[Environment] Loaded {len(df)} players")
        print(f"[Environment] Positions: {df['position'].value_counts().to_dict()}")

    def reset(self, our_team_id: Optional[int] = None):
        self.our_team_id = our_team_id if our_team_id is not None else self.rng.randint(0, self.n_teams - 1)
        self.current_round = 1
        self.draft_history = []
        self.rosters = {tid: [] for tid in range(self.n_teams)}
        self.roster_counts = {tid: {pos: 0 for pos in self.roster_limits.keys()} for tid in range(self.n_teams)}
        return self.our_team_id

    def get_draft_order(self, round_num: int):
        return list(range(self.n_teams)) if round_num % 2 == 1 else list(range(self.n_teams - 1, -1, -1))

    def get_available_players(self):
        taken = {p["player_index"] for p in self.draft_history}
        return self.player_pool[~self.player_pool.index.isin(taken)]

    def can_draft_position(self, team_id: int, position: str):
        return position in self.roster_limits and self.roster_counts[team_id][position] < self.roster_limits[position]

    def get_roster_needs(self, team_id: int):
        needs = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            needs[pos] = max(0, self.starting_requirements[pos] - self.roster_counts[team_id][pos])
        total_needed = 8
        filled = sum(self.roster_counts[team_id].values())
        needs["FLEX"] = max(0, total_needed - filled)
        return needs

    def get_context_vector(self, team_id: int, round_num: int):
        needs = self.get_roster_needs(team_id)
        roster_size = len(self.rosters[team_id])
        picks_remaining = self.n_rounds - round_num + 1
        draft_position = team_id / (self.n_teams - 1) if self.n_teams > 1 else 0.5
        order = self.get_draft_order(round_num)
        pick_in_round = order.index(team_id) + 1

        return np.array([
            round_num / self.n_rounds, pick_in_round / self.n_teams,
            needs.get("QB", 0), needs.get("RB", 0), needs.get("WR", 0),
            needs.get("TE", 0), needs.get("FLEX", 0),
            roster_size / self.n_rounds, picks_remaining / self.n_rounds, draft_position
        ], dtype=np.float32)

    def get_available_at_position(self, team_id: int, position: str):
        """Get available players at position"""
        if not self.can_draft_position(team_id, position):
            return []

        available = self.get_available_players()
        if available.empty:
            return []

        avail_pos = available[available["position"] == position]
        return [row for _, row in avail_pos.iterrows()]

    def make_pick(self, team_id: int, player_row: pd.Series):
        info = {
            "round": self.current_round, "team_id": team_id,
            "player_index": player_row.name, "position": player_row["position"],
            "first_name": player_row["first_name"], "last_name": player_row["last_name"],
            "fantasy_points_ppr": player_row["fantasy_points_ppr"],
            "adp": player_row["fantasy_adp"]
        }
        self.draft_history.append(info)
        self.rosters[team_id].append(player_row)
        if player_row["position"] in self.roster_counts[team_id]:
            self.roster_counts[team_id][player_row["position"]] += 1

    def bot_pick(self, team_id: int, slippage=15):
        """Bot that heavily prefers top ADP players.

        Default behavior:
        - Best ADP: 85%
        - Second best: 10%
        - Third best: 4%
        - All remaining (within slippage window): split 1%
        """
        available = self.get_available_players()
        if available.empty:
            return None

        # Only consider players that fit roster limits
        legal = [
            row for _, row in available.iterrows()
            if self.can_draft_position(team_id, row["position"])
        ]

        if not legal:
            return None

        legal_df = pd.DataFrame(legal).sort_values("fantasy_adp")

        # Limit to top N by ADP (slippage window)
        top_n = min(slippage, len(legal_df))
        top_players = legal_df.iloc[:top_n]

        # Build weights
        weights = np.zeros(top_n, dtype=np.float64)

        if top_n >= 1:
            weights[0] = 0.85  # best ADP
        if top_n >= 2:
            weights[1] = 0.10  # second best
        if top_n >= 3:
            weights[2] = 0.04  # third best

        if top_n > 3:
            # Remaining 1% split among indices 3..top_n-1
            remaining_prob = 1.0 - weights.sum()
            if remaining_prob < 0:
                # numerical safety (shouldn't really happen with 0.85+0.10+0.04)
                remaining_prob = 0.0
            tail_count = top_n - 3
            if tail_count > 0 and remaining_prob > 0:
                weights[3:] = remaining_prob / tail_count

        # If something weird happens (e.g., top_n < 3) and sum != 1, renormalize
        if weights.sum() == 0:
            # fallback to uniform if somehow everything is 0
            weights[:] = 1.0 / top_n
        else:
            weights /= weights.sum()

        # Sample according to these probabilities
        chosen_idx = np.random.choice(top_n, p=weights)
        return top_players.iloc[chosen_idx]

    def build_starting_lineup(self, team_id: int):
        players = self.rosters[team_id]
        lineup = {"QB": [], "RB": [], "WR": [], "TE": [], "FLEX": [], "BENCH": []}

        by_pos = {pos: [] for pos in self.roster_limits.keys()}
        for p in players:
            if p["position"] in by_pos:
                by_pos[p["position"]].append(p)

        for pos in by_pos:
            by_pos[pos].sort(key=lambda x: x["fantasy_points_ppr"], reverse=True)

        used = set()

        if by_pos["QB"]:
            lineup["QB"].append(by_pos["QB"][0])
            used.add(by_pos["QB"][0].name)

        for pos, n in [("RB", 2), ("WR", 2), ("TE", 1)]:
            for p in by_pos[pos][:n]:
                lineup[pos].append(p)
                used.add(p.name)

        flex_cands = [p for pos in ["RB", "WR", "TE"] for p in by_pos[pos] if p.name not in used]
        flex_cands.sort(key=lambda x: x["fantasy_points_ppr"], reverse=True)
        for p in flex_cands[:2]:
            lineup["FLEX"].append(p)
            used.add(p.name)

        for p in players:
            if p.name not in used:
                lineup["BENCH"].append(p)

        return lineup

    def team_score(self, team_id: int):
        lineup = self.build_starting_lineup(team_id)
        return sum(
            float(p["fantasy_points_ppr"])
            for pos in ["QB", "RB", "WR", "TE", "FLEX"]
            for p in lineup[pos]
        )

    def evaluate_league(self):
        return {tid: self.team_score(tid) for tid in range(self.n_teams)}


# ============================================================================
# HIERARCHICAL AGENT
# ============================================================================

class HierarchicalAgent:
    """Two-stage agent with position + player networks"""

    def __init__(self, position_model, player_models, env):
        self.position_model = position_model
        self.player_models = player_models
        self.env = env
        self.device = next(position_model.parameters()).device

        self.positions = ["QB", "RB", "WR", "TE"]
        self.pos_to_idx = {p: i for i, p in enumerate(self.positions)}

        print("[Agent] Hierarchical agent initialized")

    def pick_player(self, env, team_id, round_num, epsilon, training=True):
        """Two-stage pick: position then player"""

        context = env.get_context_vector(team_id, round_num)
        context_tensor = torch.tensor(context, dtype=torch.float32, device=self.device).unsqueeze(0)

        # STAGE 1: Choose position
        legal_positions = []
        legal_indices = []

        for pos in self.positions:
            if env.get_available_at_position(team_id, pos):
                legal_positions.append(pos)
                legal_indices.append(self.pos_to_idx[pos])

        if not legal_positions:
            return None, None

        self.position_model.train() if training else self.position_model.eval()

        with torch.set_grad_enabled(training):
            position_logits = self.position_model(context_tensor).squeeze(0)

            # Mask illegal positions
            legal_mask = torch.full((4,), float("-inf"), device=self.device)
            for idx in legal_indices:
                legal_mask[idx] = 0.0

            masked_logits = position_logits + legal_mask
            position_probs = torch.softmax(masked_logits, dim=0)

            # Epsilon-greedy position
            if training and random.random() < epsilon:
                chosen_pos_idx = random.choice(legal_indices)
            else:
                chosen_pos_idx = int(torch.argmax(position_probs).item())

            position_log_prob = torch.log(position_probs[chosen_pos_idx] + 1e-10) if training else None

        chosen_position = self.positions[chosen_pos_idx]

        # STAGE 2: Choose player at that position
        available_players = env.get_available_at_position(team_id, chosen_position)

        if not available_players:
            return None, None

        # FIXED: Extract features properly using feature extraction functions
        player_features = []

        for player_row in available_players:
            if chosen_position == "QB":
                feats = get_qb_features(player_row, env.feature_means, env.feature_stds)
            elif chosen_position == "RB":
                feats = get_rb_features(player_row, env.feature_means, env.feature_stds)
            elif chosen_position == "WR":
                feats = get_wr_features(player_row, env.feature_means, env.feature_stds)
            else:  # TE
                feats = get_te_features(player_row, env.feature_means, env.feature_stds)

            player_features.append(feats)

        player_tensor = torch.tensor(np.stack(player_features), dtype=torch.float32, device=self.device)
        context_batch = context_tensor.repeat(len(available_players), 1)

        # Get player values
        player_model = self.player_models[chosen_position]
        player_model.train() if training else player_model.eval()

        with torch.set_grad_enabled(training):
            player_values = player_model(player_tensor, context_batch).squeeze(1)
            player_probs = torch.softmax(player_values, dim=0)

            # Epsilon-greedy player
            if training and random.random() < epsilon:
                player_idx = random.randrange(len(available_players))
            else:
                player_idx = int(torch.argmax(player_probs).item())

            player_log_prob = torch.log(player_probs[player_idx] + 1e-10) if training else None

        chosen_player = available_players[player_idx]

        # Combine log probs
        total_log_prob = None
        if training and position_log_prob is not None and player_log_prob is not None:
            total_log_prob = position_log_prob + player_log_prob

        return chosen_player, total_log_prob


# ============================================================================
# RESTART DETECTION (currently unused, but kept for future use)
# ============================================================================

def should_restart(scores, rank_threshold=8.0):
    """
    Restart if average rank > threshold over last 200 episodes

    Args:
        scores: All episode scores
        rank_threshold: Restart if avg rank worse than this (8 = bottom third)
    """

    if len(scores) < 400:
        return False, None

    # We need to track ranks, not scores - but for now use score as proxy
    # TODO: Track actual ranks during training

    recent_200 = scores[-200:]
    previous_200 = scores[-400:-200]

    recent_avg = np.mean(recent_200)
    prev_avg = np.mean(previous_200)
    improvement = recent_avg - prev_avg

    # Restart if not improving
    if improvement < 5.0:
        return True, f"Stalled: {recent_avg:.1f} (only {improvement:+.1f} improvement)"

    return False, None


# ============================================================================
# TRAINING
# ============================================================================

def train(total_episodes=5000, learning_rate=1e-3):
    """Train hierarchical agent WITHOUT restarts - let it learn!"""

    print("=" * 80)
    print("HIERARCHICAL RL (NO RESTARTS)")
    print("=" * 80)
    print(f"\nImprovements:")
    print(f"  ✓ Fixed feature extraction")
    print(f"  ✓ Bot slippage (top 15)")
    print(f"  ✓ Fast epsilon decay")
    print(f"  ✓ NO RESTARTS - let it learn!")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_scores = []
    all_ranks = []

    best_avg_score = -float('inf')
    best_models = None

    # Create environment ONCE
    env = DraftEnvironment(
        condensed_path="nfl_players_condensed.csv",
        stats_path="nfl_players_2024_stats.csv",
        n_teams=12,
        n_rounds=8,
        seed=42
    )

    # Create networks ONCE
    position_model = PositionPolicyNetwork(context_size=10).to(device)

    # Get feature sizes
    sample_qb = env.player_pool[env.player_pool["position"] == "QB"].iloc[0]
    sample_rb = env.player_pool[env.player_pool["position"] == "RB"].iloc[0]
    sample_wr = env.player_pool[env.player_pool["position"] == "WR"].iloc[0]
    sample_te = env.player_pool[env.player_pool["position"] == "TE"].iloc[0]

    qb_size = len(get_qb_features(sample_qb, env.feature_means, env.feature_stds))
    rb_size = len(get_rb_features(sample_rb, env.feature_means, env.feature_stds))
    wr_size = len(get_wr_features(sample_wr, env.feature_means, env.feature_stds))
    te_size = len(get_te_features(sample_te, env.feature_means, env.feature_stds))

    print(f"Feature sizes: QB={qb_size}, RB={rb_size}, WR={wr_size}, TE={te_size}")

    player_models = {
        "QB": PlayerValueNetwork(qb_size, context_size=10).to(device),
        "RB": PlayerValueNetwork(rb_size, context_size=10).to(device),
        "WR": PlayerValueNetwork(wr_size, context_size=10).to(device),
        "TE": PlayerValueNetwork(te_size, context_size=10).to(device),
    }

    # Optimizers
    position_optimizer = optim.Adam(position_model.parameters(), lr=learning_rate)
    player_optimizers = {
        pos: optim.Adam(model.parameters(), lr=learning_rate)
        for pos, model in player_models.items()
    }

    agent = HierarchicalAgent(position_model, player_models, env)

    print("\nTraining without restarts...\n")

    for episodes_done in range(total_episodes):

        # Current epsilon schedule
        epsilon = max(0.05, 0.5 - 0.45 * min(1.0, episodes_done / 1000))

        our_team = env.reset()
        log_probs = []

        # -----------------------------
        # Run draft
        # -----------------------------
        for rnd in range(1, 9):
            env.current_round = rnd
            order = env.get_draft_order(rnd)

            for tid in order:
                if tid == our_team:
                    player, log_prob = agent.pick_player(env, tid, rnd, epsilon, training=True)
                    if player is not None:
                        env.make_pick(tid, player)
                        if log_prob is not None:
                            log_probs.append(log_prob)
                else:
                    player = env.bot_pick(tid)
                    if player is not None:
                        env.make_pick(tid, player)

        # -----------------------------
        # Evaluate
        # -----------------------------
        scores_dict = env.evaluate_league()
        our_score = scores_dict[our_team]
        league_scores = list(scores_dict.values())
        our_rank = sorted(league_scores, reverse=True).index(our_score) + 1

        # Reward = z-score of final fantasy score
        league_mean = np.mean(league_scores)
        league_std = np.std(league_scores) if np.std(league_scores) > 0 else 1.0
        reward = (our_score - league_mean) / league_std

        # -----------------------------
        # Backprop
        # -----------------------------
        if log_probs:
            loss = -(torch.stack(log_probs) * reward).mean()

            position_optimizer.zero_grad()
            for opt in player_optimizers.values():
                opt.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(position_model.parameters(), 1.0)
            for model in player_models.values():
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            position_optimizer.step()
            for opt in player_optimizers.values():
                opt.step()

        # -----------------------------
        # Tracking + Logging
        # -----------------------------
        all_scores.append(our_score)
        all_ranks.append(our_rank)

        if (episodes_done + 1) % 50 == 0:
            window = min(100, len(all_scores))
            avg_score = np.mean(all_scores[-window:])
            avg_rank = np.mean(all_ranks[-window:])

            print(
                f"Ep {episodes_done+1:4d}/{total_episodes} | "
                f"Score:{our_score:6.1f} Rank:{our_rank:2d}/12 | "
                f"Avg{window}[Sc:{avg_score:6.1f} Rk:{avg_rank:4.1f}] | "
                f"ε:{epsilon:.3f}"
            )

        # Save best models
        if len(all_scores) >= 100:
            recent_avg = np.mean(all_scores[-100:])
            if recent_avg > best_avg_score:
                best_avg_score = recent_avg
                best_models = {
                    "position": copy.deepcopy(position_model.state_dict()),
                    "players": {
                        pos: copy.deepcopy(model.state_dict())
                        for pos, model in player_models.items()
                    }
                }

    # Load best models
    if best_models:
        position_model.load_state_dict(best_models["position"])
        for pos, state in best_models["players"].items():
            player_models[pos].load_state_dict(state)

    # Save
    torch.save({
        "position": position_model.state_dict(),
        "players": {pos: model.state_dict() for pos, model in player_models.items()}
    }, "hierarchical_agent_final.pth")

    print(f"\n✓ Saved to hierarchical_agent_final.pth")

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Episodes: {total_episodes}")
    print(f"Avg score: {np.mean(all_scores):.1f}")
    print(f"Best 100-ep avg: {best_avg_score:.1f}")
    print(f"Avg rank: {np.mean(all_ranks):.2f} / 12")
    print(f"Final 500 avg score: {np.mean(all_scores[-500:]):.1f}")
    print(f"Final 500 avg rank: {np.mean(all_ranks[-500:]):.2f}")

    return position_model, player_models, env, all_scores, all_ranks


def plot_results(scores, ranks):
    """Quick visualization"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(scores, alpha=0.3, linewidth=0.5)
    if len(scores) >= 100:
        ma = pd.Series(scores).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label='100-Ep MA')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Training Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ranks, alpha=0.3, linewidth=0.5)
    if len(ranks) >= 100:
        ma = pd.Series(ranks).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label='100-Ep MA')
    ax.axhline(y=6.5, linestyle='--', label='Random')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rank (1=Best)")
    ax.set_title("Training Ranks")
    ax.set_ylim(12.5, 0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("hierarchical_training.png", dpi=150)
    print("✓ Saved to hierarchical_training.png")
    plt.close()


# ============================================================================
# POLICY INSPECTION / REPORTING
# ============================================================================

def inspect_learned_policy(position_model, player_models, env):
    """
    Print a human-readable summary of the learned position policy and
    player preferences, similar to the fantasy_draft_agent diagnostics.
    """
    device = next(position_model.parameters()).device
    position_model.eval()
    for m in player_models.values():
        m.eval()

    print()
    print("=" * 80)
    print("HIERARCHICAL RL - Position + Player Policies")
    print("=" * 80)
    print()
    print("Architecture:")
    print("  • Position network: decides QB/RB/WR/TE")
    print("  • Player networks: 4 separate networks for each position")
    print("  • Learns which specific players to target, not just positions")
    print("=" * 80)
    print()

    # ----------------------------------------------------------------------
    # LEARNED POSITION POLICY - by round for a mid-slot team (static context)
    # ----------------------------------------------------------------------
    mid_team = env.n_teams // 2  # e.g., 6 for 12-team league
    env.reset(our_team_id=mid_team)

    positions = ["QB", "RB", "WR", "TE"]

    print("=" * 80)
    print("LEARNED POSITION POLICY (probabilities by round, mid-slot team)")
    print("=" * 80)

    with torch.no_grad():
        for rnd in range(1, env.n_rounds + 1):
            context = env.get_context_vector(mid_team, rnd)
            ctx = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
            logits = position_model(ctx).squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

            qb_p, rb_p, wr_p, te_p = probs
            print(
                f"Round {rnd}: QB={qb_p:0.2f}, RB={rb_p:0.2f}, "
                f"WR={wr_p:0.2f}, TE={te_p:0.2f}"
            )

    print()
    print("=" * 80)
    print("LEARNED PLAYER PREFERENCES (top players per position)")
    print("=" * 80)
    print()

    # ----------------------------------------------------------------------
    # LEARNED PLAYER PREFERENCES - top 10 per position
    # ----------------------------------------------------------------------
    def eval_player_value(row, pos: str):
        if pos == "QB":
            feats = get_qb_features(row, env.feature_means, env.feature_stds)
        elif pos == "RB":
            feats = get_rb_features(row, env.feature_means, env.feature_stds)
        elif pos == "WR":
            feats = get_wr_features(row, env.feature_means, env.feature_stds)
        else:
            feats = get_te_features(row, env.feature_means, env.feature_stds)

        feats_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        # Use a neutral-ish context: mid team, middle round (e.g., round 4)
        context = env.get_context_vector(mid_team, round_num=min(4, env.n_rounds))
        ctx_t = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            val = player_models[pos](feats_t, ctx_t).item()
        return val

    for pos in positions:
        subset = env.player_pool[env.player_pool["position"] == pos]
        if subset.empty:
            continue

        player_vals = []
        for idx, row in subset.iterrows():
            val = eval_player_value(row, pos)
            player_vals.append((row, val))

        # Sort descending by learned value
        player_vals.sort(key=lambda x: x[1], reverse=True)

        print(f"Top 10 {pos}s according to the learned player policy:")
        for rank, (row, val) in enumerate(player_vals[:10], start=1):
            name = f"{row['first_name']} {row['last_name']}"
            adp = row['fantasy_adp']
            print(
                f"  {rank:2d}. {name:<24} value={val:6.3f}  ADP={adp:6.1f}"
            )
        print()

    # ----------------------------------------------------------------------
    # POSITION POLICY MATRIX: round × pick × position (static, empty rosters)
    # ----------------------------------------------------------------------
    print("[STEP] Computing position policy matrix (round × pick × position)...")
    print()
    print("=" * 80)
    print("LEARNED POSITION POLICY BY ROUND AND PICK (STATIC, EMPTY ROSTERS)")
    print("Rows = rounds, Columns = picks in that round (snake-order index)")
    print("Each line shows: QB/RB/WR/TE probabilities.")
    print("=" * 80)
    print()

    # Reset to empty rosters for a clean static view
    env.reset()

    with torch.no_grad():
        for rnd in range(1, env.n_rounds + 1):
            order = env.get_draft_order(rnd)
            print(f"Round {rnd}:")
            for pick_idx, team_id in enumerate(order, start=1):
                context = env.get_context_vector(team_id, rnd)
                ctx = torch.tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
                logits = position_model(ctx).squeeze(0)
                probs = torch.softmax(logits, dim=0).cpu().numpy()
                qb_p, rb_p, wr_p, te_p = probs

                print(
                    f"  Pick {pick_idx:2d}: QB={qb_p:0.2f}, RB={rb_p:0.2f}, "
                    f"WR={wr_p:0.2f}, TE={te_p:0.2f}"
                )
            print()

    print("=" * 80)
    print("✓ HIERARCHICAL RL TRAINING COMPLETE!")
    print("=" * 80)


def main():
    """Run fixed hierarchical training"""

    import os

    if not os.path.exists("nfl_players_condensed.csv"):
        print("ERROR: Missing nfl_players_condensed.csv")
        return

    print("\n" + "=" * 80)
    print("HIERARCHICAL RL - FIXED VERSION")
    print("=" * 80)

    print("\n[STEP 1/3] Training...")
    position_model, player_models, env, scores, ranks = train(
        total_episodes=5000,
        learning_rate=1e-3
    )

    print("\n[STEP 2/3] Plotting training curves...")
    plot_results(scores, ranks)

    print("\n[STEP 3/3] Inspecting learned policy (positions + players)...\n")
    inspect_learned_policy(position_model, player_models, env)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    main()
