"""
HIERARCHICAL RL POLICY

1. Position Network: Decides which position to draft (QB/RB/WR/TE)
2. Player Networks: Four separate networks (one per position) that evaluate
   specific players at that position
   - Each position has its own file for customization
"""

import copy
import random
from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import os
import time

# Import each position's network from separate files
from qb_network import QBPlayerNetwork, get_qb_features
from rb_network import RBPlayerNetwork, get_rb_features
from wr_network import WRPlayerNetwork, get_wr_features
from te_network import TEPlayerNetwork, get_te_features


# ============================================================================
# NETWORKS
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
            nn.Linear(32, 4),  # Output: scores for QB/RB/WR/TE
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [batch, context_size]
        returns: [batch, 4] scores for each position
        """
        return self.network(context)


# ============================================================================
# ENVIRONMENT
# ============================================================================


class DraftEnvironment:
    def __init__(
        self,
        draft_data_path: str,
        scoring_data_path: str,
        n_teams: int = 12,
        n_rounds: int = 8,
        seed: Optional[int] = None,
    ):
        self.n_teams = n_teams
        self.n_rounds = n_rounds

        if seed is not None:
            self.rng = random.Random(seed)
            print(f"[Env] Using env RNG seed: {seed}")
        else:
            self.rng = random.Random()
            print("[Env] Using unseeded env RNG")

        self.roster_limits = {"QB": 2, "RB": 5, "WR": 5, "TE": 3}
        self.starting_requirements = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}

        # Load condensed CSV for drafting (with ADP and features)
        df_draft = pd.read_csv(draft_data_path)

        # Convert position one-hot columns to a single 'position' column
        if "pos_qb" in df_draft.columns:
            df_draft["position"] = df_draft.apply(self._get_position_from_onehot, axis=1)

        # Filter and sort by ADP
        df_draft = df_draft[df_draft["position"].isin(["QB", "RB", "WR", "TE"])].copy()
        df_draft = df_draft[df_draft["fantasy_adp"] < 300].copy()
        df_draft = df_draft.sort_values("fantasy_adp").reset_index(drop=True)

        # Load actual 2024 stats for scoring
        df_scoring = pd.read_csv(scoring_data_path)
        df_scoring = df_scoring[df_scoring["position"].isin(["QB", "RB", "WR", "TE"])].copy()

        # Merge scoring data into draft data
        df_draft = df_draft.merge(
            df_scoring[["first_name", "last_name", "fantasy_points_ppr"]],
            on=["first_name", "last_name"],
            how="left",
            suffixes=("", "_actual"),
        )

        # Use actual 2024 fantasy_points_ppr for scoring
        df_draft['fantasy_points_ppr'] = df_draft['fantasy_points_ppr'].fillna(0.0)

        self.player_pool = df_draft
        
        # Define which columns to use as player features
        self.feature_columns = [
            'fantasy_adp', 'age', 'is_rookie', 'team_win_pct_weighted',
            'avg_points_per_game', 'avg_games', 'avg_rushing_yards',
            'avg_rushing_tds', 'avg_receptions', 'avg_receiving_yards',
            'avg_receiving_tds', 'avg_passing_yards', 'avg_passing_tds'
        ]
        
        # Normalize features for neural network
        self._normalize_features()
        
        self.reset()
        print(f"[Environment] Loaded {len(df_draft)} players for drafting")
        print(f"[Environment] Using actual 2024 stats for scoring")

    def _normalize_features(self):
        """Normalize feature columns to [0, 1] range"""
        for col in self.feature_columns:
            if col in self.player_pool.columns:
                col_data = self.player_pool[col].fillna(0.0)
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val > min_val:
                    self.player_pool[f'{col}_norm'] = (col_data - min_val) / (max_val - min_val)
                else:
                    self.player_pool[f'{col}_norm'] = 0.0

    def get_player_features(self, player_row: pd.Series, position: str) -> np.ndarray:
        """
        Extract position-specific feature vector for a player
        Each position uses its own feature extraction function
        """
        if position == "QB":
            features = get_qb_features(player_row, self)
        elif position == "RB":
            features = get_rb_features(player_row, self)
        elif position == "WR":
            features = get_wr_features(player_row, self)
        elif position == "TE":
            features = get_te_features(player_row, self)
        else:
            features = []
        
        return np.array(features, dtype=np.float32)

    def _get_position_from_onehot(self, row):
        """Convert one-hot position columns to single position string"""
        if row.get("pos_qb", 0) == 1:
            return "QB"
        elif row.get("pos_rb", 0) == 1:
            return "RB"
        elif row.get("pos_wr", 0) == 1:
            return "WR"
        elif row.get("pos_te", 0) == 1:
            return "TE"
        return "UNKNOWN"

    def reset(self, our_team_id: Optional[int] = None):
        self.our_team_id = (
            our_team_id if our_team_id is not None else self.rng.randint(0, self.n_teams - 1)
        )
        self.current_round = 1
        self.draft_history = []
        self.rosters = {tid: [] for tid in range(self.n_teams)}
        self.roster_counts = {
            tid: {pos: 0 for pos in self.roster_limits.keys()} for tid in range(self.n_teams)
        }
        return self.our_team_id

    def get_draft_order(self, round_num: int):
        return (
            list(range(self.n_teams))
            if round_num % 2 == 1
            else list(range(self.n_teams - 1, -1, -1))
        )

    def get_available_players(self):
        taken = {p["player_index"] for p in self.draft_history}
        return self.player_pool[~self.player_pool.index.isin(taken)]

    def can_draft_position(self, team_id: int, position: str):
        return (
            position in self.roster_limits
            and self.roster_counts[team_id][position] < self.roster_limits[position]
        )

    def get_roster_needs(self, team_id: int):
        needs = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            needs[pos] = max(
                0, self.starting_requirements[pos] - self.roster_counts[team_id][pos]
            )
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

        return np.array(
            [
                round_num / self.n_rounds,
                pick_in_round / self.n_teams,
                needs.get("QB", 0),
                needs.get("RB", 0),
                needs.get("WR", 0),
                needs.get("TE", 0),
                needs.get("FLEX", 0),
                roster_size / self.n_rounds,
                picks_remaining / self.n_rounds,
                draft_position,
            ],
            dtype=np.float32,
        )

    def make_pick(self, team_id: int, player_row: pd.Series):
        info = {
            "round": self.current_round,
            "team_id": team_id,
            "player_index": player_row.name,
            "position": player_row["position"],
            "first_name": player_row["first_name"],
            "last_name": player_row["last_name"],
            "fantasy_points_ppr": player_row["fantasy_points_ppr"],
            "adp": player_row["fantasy_adp"],
        }
        self.draft_history.append(info)
        self.rosters[team_id].append(player_row)
        if player_row["position"] in self.roster_counts[team_id]:
            self.roster_counts[team_id][player_row["position"]] += 1

    def bot_pick(self, team_id: int):
        available = self.get_available_players()
        if available.empty:
            return None
        legal = [
            row
            for _, row in available.iterrows()
            if self.can_draft_position(team_id, row["position"])
        ]
        if not legal:
            return None
        return pd.DataFrame(legal).sort_values("fantasy_adp").iloc[0]

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

        flex_cands = [
            p
            for pos in ["RB", "WR", "TE"]
            for p in by_pos[pos]
            if p.name not in used
        ]
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

    def get_available_at_position(self, team_id: int, position: str):
        """Get all available players at a position that this team can draft"""
        if not self.can_draft_position(team_id, position):
            return []
        
        available = self.get_available_players()
        if available.empty:
            return []
        
        avail_pos = available[available["position"] == position]
        return [row for _, row in avail_pos.iterrows()]

    def evaluate_league(self):
        return {tid: self.team_score(tid) for tid in range(self.n_teams)}


# ============================================================================
# HIERARCHICAL AGENT
# ============================================================================


class HierarchicalDraftAgent:
    """
    Two-stage decision process:
    1. Position policy: decides which position to draft
    2. Player policy: decides which specific player at that position
    """

    def __init__(self, position_model: nn.Module, player_models: Dict[str, nn.Module]):
        self.position_model = position_model
        self.player_models = player_models  # {"QB": model, "RB": model, ...}
        self.device = next(position_model.parameters()).device
        
        self.positions = ["QB", "RB", "WR", "TE"]
        self.pos_to_idx = {p: i for i, p in enumerate(self.positions)}

    def pick_player(self, env, team_id: int, round_num: int, epsilon: float, training: bool = True):
        """
        Two-stage pick:
        1. Choose position using position_model
        2. Choose specific player at that position using player_models[position]
        """
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
            return None, None, None
        
        # Get position scores
        if training:
            self.position_model.train()
        else:
            self.position_model.eval()
        
        with torch.set_grad_enabled(training):
            position_logits = self.position_model(context_tensor).squeeze(0)  # [4]
            
            # Mask illegal positions
            legal_mask = torch.full((4,), float('-inf'), device=self.device)
            for idx in legal_indices:
                legal_mask[idx] = 0.0
            
            masked_logits = position_logits + legal_mask
            position_probs = torch.softmax(masked_logits, dim=0)
            
            # ε-greedy position selection
            if training and random.random() < epsilon:
                chosen_pos_idx = random.choice(legal_indices)
            else:
                chosen_pos_idx = int(torch.argmax(position_probs).item())
            
            position_log_prob = torch.log(position_probs[chosen_pos_idx] + 1e-10) if training else None
        
        chosen_position = self.positions[chosen_pos_idx]
        
        # STAGE 2: Choose specific player at that position
        available_players = env.get_available_at_position(team_id, chosen_position)
        
        if not available_players:
            return None, None, None
        
        # Get player features for all available players at this position
        player_features = []
        for player_row in available_players:
            feat = env.get_player_features(player_row, chosen_position)
            player_features.append(feat)
        
        player_features_tensor = torch.tensor(
            np.stack(player_features),
            dtype=torch.float32,
            device=self.device
        )
        
        # Repeat context for each player
        context_batch = context_tensor.repeat(len(available_players), 1)
        
        # Get player values from position-specific model
        player_model = self.player_models[chosen_position]
        
        if training:
            player_model.train()
        else:
            player_model.eval()
        
        with torch.set_grad_enabled(training):
            player_values = player_model(player_features_tensor, context_batch).squeeze(1)
            player_probs = torch.softmax(player_values, dim=0)
            
            # ε-greedy player selection
            if training and random.random() < epsilon:
                player_idx = random.randrange(len(available_players))
            else:
                player_idx = int(torch.argmax(player_probs).item())
            
            player_log_prob = torch.log(player_probs[player_idx] + 1e-10) if training else None
        
        chosen_player = available_players[player_idx]
        
        # Combine log probs for total action probability
        total_log_prob = None
        if training and position_log_prob is not None and player_log_prob is not None:
            total_log_prob = position_log_prob + player_log_prob
        
        return chosen_player, total_log_prob, chosen_position


# ============================================================================
# TRAINING
# ============================================================================


def train_hierarchical_agent(
    total_episodes=5000,
    learning_rate=1e-3,
):
    """Train hierarchical agent with position + player policies"""

    print("=" * 80)
    print("HIERARCHICAL RL - Position + Player Policies")
    print("=" * 80)
    print("\nArchitecture:")
    print("  • Position network: decides QB/RB/WR/TE")
    print("  • Player networks: 4 separate networks for each position")
    print("  • Learns which specific players to target, not just positions")
    print("=" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a derived env seed so runs are reproducible given the base seed,
    # but different across separate runs when the base seed changes.
    env_seed = random.randint(0, 1_000_000_000)
    print(f"[SEED] Using env seed: {env_seed}")
    env = DraftEnvironment(
        draft_data_path="nfl_players_condensed.csv",
        scoring_data_path="nfl_players_2024_stats.csv",
        n_teams=12,
        n_rounds=8,
        seed=env_seed,
    )

    # Create networks - each position has its own architecture and features
    position_model = PositionPolicyNetwork(context_size=10).to(device)
    
    # Get feature sizes for each position (may be different!)
    sample_qb = env.player_pool[env.player_pool['position'] == 'QB'].iloc[0]
    sample_rb = env.player_pool[env.player_pool['position'] == 'RB'].iloc[0]
    sample_wr = env.player_pool[env.player_pool['position'] == 'WR'].iloc[0]
    sample_te = env.player_pool[env.player_pool['position'] == 'TE'].iloc[0]
    
    qb_feature_size = len(get_qb_features(sample_qb, env))
    rb_feature_size = len(get_rb_features(sample_rb, env))
    wr_feature_size = len(get_wr_features(sample_wr, env))
    te_feature_size = len(get_te_features(sample_te, env))
    
    print(f"Feature sizes: QB={qb_feature_size}, RB={rb_feature_size}, WR={wr_feature_size}, TE={te_feature_size}")
    
    player_models = {
        "QB": QBPlayerNetwork(qb_feature_size, context_size=10).to(device),
        "RB": RBPlayerNetwork(rb_feature_size, context_size=10).to(device),
        "WR": WRPlayerNetwork(wr_feature_size, context_size=10).to(device),
        "TE": TEPlayerNetwork(te_feature_size, context_size=10).to(device),
    }
    
    # Separate optimizers for each network
    position_optimizer = optim.Adam(position_model.parameters(), lr=learning_rate)
    player_optimizers = {
        pos: optim.Adam(model.parameters(), lr=learning_rate)
        for pos, model in player_models.items()
    }
    
    agent = HierarchicalDraftAgent(position_model, player_models)

    all_scores = []
    all_ranks = []
    all_rewards = []

    best_avg_score = -float("inf")
    best_models = None

    for ep in range(total_episodes):
        epsilon = max(0.05, 0.9 - 0.85 * (ep / total_episodes))

        our_team = env.reset()
        log_probs = []

        for rnd in range(1, 9):
            env.current_round = rnd
            order = env.get_draft_order(rnd)

            for tid in order:
                if tid == our_team:
                    player, log_prob, position = agent.pick_player(env, tid, rnd, epsilon, training=True)
                    if player is not None:
                        env.make_pick(tid, player)
                        if log_prob is not None:
                            log_probs.append(log_prob)
                else:
                    player = env.bot_pick(tid)
                    if player is not None:
                        env.make_pick(tid, player)

        scores_dict = env.evaluate_league()
        our_score = scores_dict[our_team]
        league_scores = list(scores_dict.values())
        our_rank = sorted(league_scores, reverse=True).index(our_score) + 1

        league_mean = np.mean(league_scores)
        league_std = np.std(league_scores) if np.std(league_scores) > 0 else 1.0
        reward = (our_score - league_mean) / league_std

        if log_probs:
            loss = -(torch.stack(log_probs) * reward).mean()

            # Update all networks
            position_optimizer.zero_grad()
            for opt in player_optimizers.values():
                opt.zero_grad()
            
            loss.backward()
            
            nn.utils.clip_grad_norm_(position_model.parameters(), 5.0)
            for model in player_models.values():
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            position_optimizer.step()
            for opt in player_optimizers.values():
                opt.step()

        all_scores.append(our_score)
        all_ranks.append(our_rank)
        all_rewards.append(reward)

        if (ep + 1) % 50 == 0:
            window = min(100, len(all_scores))
            avg_score = np.mean(all_scores[-window:])
            avg_rank = np.mean(all_ranks[-window:])

            print(
                f"Ep {ep+1:4d}/{total_episodes} | "
                f"Score:{our_score:6.1f} Rank:{our_rank:2d}/12 | "
                f"Avg{window}[S:{avg_score:6.1f} Rk:{avg_rank:4.1f}] | "
                f"ε:{epsilon:.3f}"
            )

        if len(all_scores) >= 100:
            recent_avg = np.mean(all_scores[-100:])
            if recent_avg > best_avg_score:
                best_avg_score = recent_avg
                best_models = {
                    'position': copy.deepcopy(position_model.state_dict()),
                    'players': {pos: copy.deepcopy(model.state_dict()) 
                               for pos, model in player_models.items()}
                }

    print("\nTraining complete.")

    if best_models is not None:
        position_model.load_state_dict(best_models['position'])
        for pos, state in best_models['players'].items():
            player_models[pos].load_state_dict(state)
        
        torch.save({
            'position_model': position_model.state_dict(),
            'player_models': {pos: model.state_dict() for pos, model in player_models.items()}
        }, "hierarchical_draft_agent.pth")
        print("\n✓ Saved best-performing models to hierarchical_draft_agent.pth")

    return position_model, player_models, all_scores, all_ranks, all_rewards


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_training(scores, ranks, rewards):
    """Plot training curves"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scores
    ax = axes[0, 0]
    ax.plot(scores, alpha=0.3)
    if len(scores) >= 100:
        ma = pd.Series(scores).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Episode Scores")
    ax.legend()

    # Ranks
    ax = axes[0, 1]
    ax.plot(ranks, alpha=0.3)
    if len(ranks) >= 100:
        ma = pd.Series(ranks).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rank (lower is better)")
    ax.set_title("Episode Ranks")
    ax.legend()

    # Rewards
    ax = axes[1, 0]
    ax.plot(rewards, alpha=0.3)
    if len(rewards) >= 100:
        ma = pd.Series(rewards).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (z-score)")
    ax.set_title("Episode Rewards")
    ax.legend()

    # Histogram of scores
    ax = axes[1, 1]
    ax.hist(scores, bins=30, alpha=0.7)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Scores")

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Train hierarchical agent"""

    print("\n" + "=" * 80)
    print("HIERARCHICAL RL - Position + Player Policies")
    print("=" * 80)

    print("\n[STEP 1/2] Training")
    print("This will take a bit for 5000 episodes...\n")

    position_model, player_models, scores, ranks, rewards = train_hierarchical_agent(
        total_episodes=5000,
        learning_rate=1e-3,
    )

    print("\n[STEP 2/2] Plotting training curves...")
    plot_training(scores, ranks, rewards)

    print("\n" + "=" * 80)
    print("✓ HIERARCHICAL RL TRAINING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    # Optional: allow an explicit base seed via env var
    env_seed_var = os.environ.get("FF_SEED")
    if env_seed_var is not None:
        base_seed = int(env_seed_var)
        print(f"[SEED] Using provided base seed from FF_SEED: {base_seed}")
    else:
        base_seed = int(time.time_ns() % (2**31 - 1))
        print(f"[SEED] Using time-based base seed: {base_seed}")

    # Deterministic behavior within this run
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)
    random.seed(base_seed)

    main()
