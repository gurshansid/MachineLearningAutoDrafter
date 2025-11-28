"""
FIXED HIERARCHICAL RL - All 5 Networks in One File

Fixes:
1. Feature extraction (was returning all zeros!)
2. Proper normalization handling
3. Correct column names from your data
4. Added restart logic
5. Bot slippage for easier learning
6. Bots + agent respect roster shape:
   - EXACTLY 8 players per team
   - Must end with at least: 1 QB, 2 RB, 2 WR, 1 TE
   - Remaining 2 spots are FLEX from RB/WR/TE
   - QBs are hard-capped at 1 → no second QB / no bench
   - Bots still pick by ADP with slippage
   - No extra "strategy" for bots beyond ADP + constraints
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

    feature_cols = [
        "fantasy_adp",
        "avg_points_per_game",
        "avg_passing_yards",
        "avg_passing_tds",
        "avg_rushing_yards",  # QB rushing matters
        "avg_rushing_tds",
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
        "fantasy_adp",
        "avg_points_per_game",
        "avg_rushing_yards",
        "avg_rushing_tds",
        "avg_receptions",  # PPR crucial for RBs
        "avg_receiving_yards",
        "avg_receiving_tds",
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
        "fantasy_adp",
        "avg_points_per_game",
        "avg_targets",  # Target share is key
        "avg_receptions",
        "avg_receiving_yards",
        "avg_receiving_tds",
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
        "fantasy_adp",
        "avg_points_per_game",
        "avg_targets",
        "avg_receptions",
        "avg_receiving_yards",
        "avg_receiving_tds",
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
    def __init__(
        self,
        condensed_path: str,
        stats_path: str,
        n_teams: int = 12,
        n_rounds: int = 8,
        seed: int = 42,
    ):
        self.n_teams = n_teams
        self.n_rounds = n_rounds
        self.rng = random.Random(seed)

        # Hard roster caps:
        # - QB is hard-capped at 1  -> no second QB, no QB bench
        # - RB/WR/TE caps are generous, but feasibility + total=8 ensure no bench
        self.roster_limits = {"QB": 1, "RB": 5, "WR": 5, "TE": 3}

        # Desired minimums for final roster (hard constraints):
        # 1 QB, 2 RB, 2 WR, 1 TE; total players = 8, so 2 more are FLEX from RB/WR/TE
        self.min_requirements = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
        self.max_players_per_team = 8

        # For context features / "needs"
        self.starting_requirements = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2}

        # Load condensed (has features for evaluation)
        df_condensed = pd.read_csv(condensed_path)

        # Convert position one-hot to single column
        if "pos_qb" in df_condensed.columns:
            df_condensed["position"] = "UNKNOWN"
            df_condensed.loc[df_condensed["pos_qb"] == 1, "position"] = "QB"
            df_condensed.loc[df_condensed["pos_rb"] == 1, "position"] = "RB"
            df_condensed.loc[df_condensed["pos_wr"] == 1, "position"] = "WR"
            df_condensed.loc[df_condensed["pos_te"] == 1, "position"] = "TE"

        # Load 2024 stats (has actual fantasy points for scoring)
        df_2024 = pd.read_csv(stats_path)

        # Merge to get actual 2024 fantasy points
        df = df_condensed.merge(
            df_2024[["first_name", "last_name", "fantasy_points_ppr"]],
            on=["first_name", "last_name"],
            how="inner",  # Only keep players in both files
            suffixes=("_pred", "_actual"),
        )

        # Use actual 2024 points for scoring
        if "fantasy_points_ppr_actual" in df.columns:
            df["fantasy_points_ppr"] = df["fantasy_points_ppr_actual"]

        # Filter to draftable
        df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
        df = df[df["fantasy_adp"] < 300].copy()
        df = df.sort_values("fantasy_adp").reset_index(drop=True)

        self.player_pool = df

        # Compute feature statistics for normalization
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.feature_means = df[numeric_cols].mean()
        self.feature_stds = df[numeric_cols].std().replace(0, 1.0)

        self.reset()
        print(f"[Environment] Loaded {len(df)} players")
        print(f"[Environment] Positions: {df['position'].value_counts().to_dict()}")

    def reset(self, our_team_id: Optional[int] = None):
        self.our_team_id = (
            our_team_id
            if our_team_id is not None
            else self.rng.randint(0, self.n_teams - 1)
        )
        self.current_round = 1
        self.draft_history = []
        self.rosters: Dict[int, list] = {tid: [] for tid in range(self.n_teams)}
        self.roster_counts: Dict[int, Dict[str, int]] = {
            tid: {pos: 0 for pos in self.roster_limits.keys()}
            for tid in range(self.n_teams)
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

    # ---- Roster needs & legality (used for BOTH agent and bots) -------------

    def get_roster_needs(self, team_id: int):
        """
        Needs based on desired lineup:
        - 1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX
        Used mainly for context features and printing; hard legality is enforced
        separately via feasibility check.
        """
        needs = {}
        for pos in ["QB", "RB", "WR", "TE"]:
            needs[pos] = max(
                0, self.starting_requirements[pos] - self.roster_counts[team_id][pos]
            )

        total_needed = self.max_players_per_team
        filled = sum(self.roster_counts[team_id].values())
        needs["FLEX"] = max(0, total_needed - filled)
        return needs

    def position_is_legal_for_roster(self, team_id: int, position: str) -> bool:
        """
        HARD legality check:

        - Total players per team is capped at 8.
        - Final roster MUST be able to end up with at least:
          1 QB, 2 RB, 2 WR, 1 TE.
        - QBs are hard-capped at 1.
        - This function checks:
          "If we draft one more player at `position` right now,
           is it STILL POSSIBLE to satisfy all minimum requirements
           with the remaining picks?"
        """
        counts = self.roster_counts[team_id]
        total = sum(counts.values())

        # Already full
        if total >= self.max_players_per_team:
            return False

        # Simulate adding one player at this position
        new_counts = counts.copy()
        new_counts[position] += 1
        new_total = total + 1
        remaining_picks = self.max_players_per_team - new_total

        # Respect hard caps (this is where QB is forced to <= 1)
        if new_counts[position] > self.roster_limits[position]:
            return False

        # Compute how many "mandatory" picks we still need after this choice
        needed = 0
        for pos, req in self.min_requirements.items():
            if new_counts[pos] < req:
                needed += (req - new_counts[pos])

        # If we don't have enough picks left to satisfy minimums, it's illegal
        if needed > remaining_picks:
            return False

        # Otherwise, this position is legal
        return True

    def can_draft_position(self, team_id: int, position: str):
        """
        Compatibility helper, just defers to hard legality.
        """
        return self.position_is_legal_for_roster(team_id, position)

    def get_context_vector(self, team_id: int, round_num: int):
        needs = self.get_roster_needs(team_id)
        roster_size = len(self.rosters[team_id])
        picks_remaining = self.n_rounds - round_num + 1
        draft_position = (
            team_id / (self.n_teams - 1) if self.n_teams > 1 else 0.5
        )
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

    def get_available_at_position(self, team_id: int, position: str):
        """Get available players at position, respecting HARD roster shape."""
        if not self.position_is_legal_for_roster(team_id, position):
            return []

        available = self.get_available_players()
        if available.empty:
            return []

        avail_pos = available[available["position"] == position]
        return [row for _, row in avail_pos.iterrows()]

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

    def bot_pick(self, team_id: int, slippage: int = 15):
        """
        Bot that:
        - Picks the top-ADP players (with slippage window),
        - BUT only from positions that keep it possible to end with:
          1 QB, 2 RB, 2 WR, 1 TE in a total of 8 players.
        - QBs are capped at 1, so bots never take a second QB.
        - No extra strategy beyond that.
        """
        # Already has a full roster under our hard constraints
        if len(self.rosters[team_id]) >= self.max_players_per_team:
            return None

        available = self.get_available_players()
        if available.empty:
            return None

        # Only consider players that are legal given hard feasibility
        legal_rows = []
        for _, row in available.iterrows():
            pos = row["position"]
            if self.position_is_legal_for_roster(team_id, pos):
                legal_rows.append(row)

        if not legal_rows:
            return None

        legal_df = pd.DataFrame(legal_rows).sort_values("fantasy_adp")

        # Limit to top N by ADP (slippage window)
        top_n = min(slippage, len(legal_df))
        top_players = legal_df.iloc[:top_n]

        # Build weights (slippage pattern)
        weights = np.zeros(top_n, dtype=np.float64)

        if top_n >= 1:
            weights[0] = 0.85  # best ADP
        if top_n >= 2:
            weights[1] = 0.10  # second best
        if top_n >= 3:
            weights[2] = 0.04  # third best

        if top_n > 3:
            remaining_prob = 1.0 - weights.sum()
            if remaining_prob < 0:
                remaining_prob = 0.0
            tail_count = top_n - 3
            if tail_count > 0 and remaining_prob > 0:
                weights[3:] = remaining_prob / tail_count

        if weights.sum() == 0:
            weights[:] = 1.0 / top_n
        else:
            weights /= weights.sum()

        chosen_idx = np.random.choice(top_n, p=weights)
        return top_players.iloc[chosen_idx]

    def build_starting_lineup(self, team_id: int):
        """
        Build the final lineup with:
        - 1 QB
        - 2 RB
        - 2 WR
        - 1 TE
        - 2 FLEX (from remaining RB/WR/TE)

        With:
        - total players = 8
        - exactly 1 QB (hard cap + min)
        => RB+WR+TE = 7
        => all RB/WR/TE are starters (no bench).
        """
        players = self.rosters[team_id]
        lineup = {"QB": [], "RB": [], "WR": [], "TE": [], "FLEX": [], "BENCH": []}

        by_pos = {pos: [] for pos in self.roster_limits.keys()}
        for p in players:
            if p["position"] in by_pos:
                by_pos[p["position"]].append(p)

        for pos in by_pos:
            by_pos[pos].sort(key=lambda x: x["fantasy_points_ppr"], reverse=True)

        used = set()

        # 1 QB (there should be exactly 1)
        if by_pos["QB"]:
            lineup["QB"].append(by_pos["QB"][0])
            used.add(by_pos["QB"][0].name)

        # 2 RB, 2 WR, 1 TE
        for pos, n in [("RB", 2), ("WR", 2), ("TE", 1)]:
            for p in by_pos[pos][:n]:
                lineup[pos].append(p)
                used.add(p.name)

        # 2 FLEX from remaining RB/WR/TE
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

        # Bench = everything else (should be empty given constraints)
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
# ROSTER / LEAGUE PRINTING HELPERS
# ============================================================================

def print_agent_roster(env: DraftEnvironment, team_id: int):
    """Pretty-print the agent's roster for the current draft (all starters)."""
    lineup = env.build_starting_lineup(team_id)

    print("  Starters:")
    for pos in ["QB", "RB", "WR", "TE", "FLEX"]:
        for p in lineup[pos]:
            name = f"{p['first_name']} {p['last_name']}"
            pts = float(p["fantasy_points_ppr"])
            adp = float(p["fantasy_adp"])
            print(f"    {pos:<5} {name:<25} PPR={pts:6.1f}  ADP={adp:6.1f}")

    # Bench should always be empty now, but we keep this for safety / debugging
    if lineup["BENCH"]:
        print("  Bench:")
        for p in lineup["BENCH"]:
            name = f"{p['first_name']} {p['last_name']}"
            pts = float(p["fantasy_points_ppr"])
            adp = float(p["fantasy_adp"])
            print(f"    BENCH {name:<25} PPR={pts:6.1f}  ADP={adp:6.1f}")


def print_league_roster_counts(env: DraftEnvironment):
    """Print how many QBs/RBs/WRs/TEs each team drafted (for debugging)."""
    print("  League roster counts (current draft):")
    print("    Team |  QB  RB  WR  TE | Total")
    print("    ------------------------------")
    for tid in range(env.n_teams):
        counts = env.roster_counts[tid]
        total = sum(counts.values())
        print(
            f"     {tid:2d} |  {counts['QB']:2d} {counts['RB']:3d} {counts['WR']:3d} {counts['TE']:3d} | {total:4d}"
        )
    print()


# ============================================================================
# HIERARCHICAL AGENT
# ============================================================================

class HierarchicalAgent:
    """Two-stage agent with position + player networks"""

    def __init__(self, position_model, player_models, env: DraftEnvironment):
        self.position_model = position_model
        self.player_models = player_models
        self.env = env
        self.device = next(position_model.parameters()).device

        self.positions = ["QB", "RB", "WR", "TE"]
        self.pos_to_idx = {p: i for i, p in enumerate(self.positions)}

        print("[Agent] Hierarchical agent initialized")

    def pick_player(
        self,
        env: DraftEnvironment,
        team_id: int,
        round_num: int,
        epsilon: float,
        training: bool = True,
    ):
        """Two-stage pick: position then player, under HARD roster constraints"""

        # If we're already full, no pick
        if len(env.rosters[team_id]) >= env.max_players_per_team:
            return None, None

        context = env.get_context_vector(team_id, round_num)
        context_tensor = torch.tensor(
            context, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # STAGE 1: Choose position
        legal_positions = []
        legal_indices = []

        for pos in self.positions:
            # Uses the same hard legality and availability as bots
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

            position_log_prob = (
                torch.log(position_probs[chosen_pos_idx] + 1e-10)
                if training
                else None
            )

        chosen_position = self.positions[chosen_pos_idx]

        # STAGE 2: Choose player at that position
        available_players = env.get_available_at_position(team_id, chosen_position)

        if not available_players:
            return None, None

        # Extract features properly using feature extraction functions
        player_features = []
        for player_row in available_players:
            if chosen_position == "QB":
                feats = get_qb_features(
                    player_row, env.feature_means, env.feature_stds
                )
            elif chosen_position == "RB":
                feats = get_rb_features(
                    player_row, env.feature_means, env.feature_stds
                )
            elif chosen_position == "WR":
                feats = get_wr_features(
                    player_row, env.feature_means, env.feature_stds
                )
            else:  # TE
                feats = get_te_features(
                    player_row, env.feature_means, env.feature_stds
                )
            player_features.append(feats)

        player_tensor = torch.tensor(
            np.stack(player_features), dtype=torch.float32, device=self.device
        )
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

            player_log_prob = (
                torch.log(player_probs[player_idx] + 1e-10) if training else None
            )

        chosen_player = available_players[player_idx]

        # Combine log probs
        total_log_prob = None
        if (
            training
            and position_log_prob is not None
            and player_log_prob is not None
        ):
            total_log_prob = position_log_prob + player_log_prob

        return chosen_player, total_log_prob


# ============================================================================
# RESTART DETECTION (currently unused, but kept for future use)
# ============================================================================

def should_restart(scores, rank_threshold: float = 8.0):
    """
    Restart if average rank > threshold over last 200 episodes
    """
    if len(scores) < 400:
        return False, None

    recent_200 = scores[-200:]
    previous_200 = scores[-400:-200]

    recent_avg = np.mean(recent_200)
    prev_avg = np.mean(previous_200)
    improvement = recent_avg - prev_avg

    if improvement < 5.0:
        return True, f"Stalled: {recent_avg:.1f} (only {improvement:+.1f} improvement)"

    return False, None


# ============================================================================
# TRAINING
# ============================================================================

def train(total_episodes: int = 5000, learning_rate: float = 1e-3):
    """Train hierarchical agent WITHOUT restarts - let it learn!"""

    print("=" * 80)
    print("HIERARCHICAL RL (NO RESTARTS)")
    print("=" * 80)
    print("\nImprovements:")
    print("  ✓ Fixed feature extraction")
    print("  ✓ Bot slippage (top 15)")
    print(
        "  ✓ HARD roster constraints (exactly 8 players, min 1 QB / 2 RB / 2 WR / 1 TE, QB cap=1)"
    )
    print("  ✓ Agent uses same roster constraints as bots")
    print("  ✓ Fast epsilon decay")
    print("  ✓ NO RESTARTS - let it learn!")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_scores = []
    all_ranks = []

    best_avg_score = -float("inf")
    best_models = None

    # Create environment ONCE
    env = DraftEnvironment(
        condensed_path="nfl_players_condensed.csv",
        stats_path="nfl_players_2024_stats.csv",
        n_teams=12,
        n_rounds=8,
        seed=42,
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
        for rnd in range(1, env.n_rounds + 1):
            env.current_round = rnd
            order = env.get_draft_order(rnd)

            for tid in order:
                if tid == our_team:
                    player, log_prob = agent.pick_player(
                        env, tid, rnd, epsilon, training=True
                    )
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

            print("  Our roster this episode:")
            print_agent_roster(env, our_team)
            print()

            # Also print league-wide roster counts to see bot shapes
            print_league_roster_counts(env)

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
                    },
                }

    # Load best models
    if best_models:
        position_model.load_state_dict(best_models["position"])
        for pos, state in best_models["players"].items():
            player_models[pos].load_state_dict(state)

    # Save
    torch.save(
        {
            "position": position_model.state_dict(),
            "players": {
                pos: model.state_dict() for pos, model in player_models.items()
            },
        },
        "hierarchical_agent_final.pth",
    )

    print("\n✓ Saved to hierarchical_agent_final.pth")

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


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(scores, ranks):
    """Quick visualization"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(scores, alpha=0.3, linewidth=0.5)
    if len(scores) >= 100:
        ma = pd.Series(scores).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title("Training Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ranks, alpha=0.3, linewidth=0.5)
    if len(ranks) >= 100:
        ma = pd.Series(ranks).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")
        ax.axhline(y=6.5, linestyle="--", label="Random")
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

def inspect_learned_policy(position_model, player_models, env: DraftEnvironment):
    """
    Print a human-readable summary of the learned player preferences
    and a single showcase draft with the trained policy.
    """
    device = next(position_model.parameters()).device
    position_model.eval()
    for m in player_models.values():
        m.eval()

    positions = ["QB", "RB", "WR", "TE"]

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
    # LEARNED PLAYER PREFERENCES - top 10 per position (neutral-ish context)
    # ----------------------------------------------------------------------
    mid_team = env.n_teams // 2
    env.reset(our_team_id=mid_team)

    print("=" * 80)
    print("LEARNED PLAYER PREFERENCES (top players per position)")
    print("=" * 80)
    print()

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
        context = env.get_context_vector(mid_team, round_num=min(4, env.n_rounds))
        ctx_t = torch.tensor(
            context, dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            val = player_models[pos](feats_t, ctx_t).item()
        return val

    for pos in positions:
        subset = env.player_pool[env.player_pool["position"] == pos]
        if subset.empty:
            continue

        player_vals = []
        for _, row in subset.iterrows():
            val = eval_player_value(row, pos)
            player_vals.append((row, val))

        player_vals.sort(key=lambda x: x[1], reverse=True)

        print(f"Top 10 {pos}s according to the learned player policy:")
        for rank, (row, val) in enumerate(player_vals[:10], start=1):
            name = f"{row['first_name']} {row['last_name']}"
            adp = row["fantasy_adp"]
            print(f"  {rank:2d}. {name:<24} value={val:6.3f}  ADP={adp:6.1f}")
        print()

    # ----------------------------------------------------------------------
    # REALISTIC DRAFT TRACE
    # ----------------------------------------------------------------------
    print("=" * 80)
    print("REALISTIC POSITION POLICY DURING A SHOWCASE DRAFT (MID-SLOT TEAM)")
    print("=" * 80)
    print("Showing masked position probabilities for our picks (ε=0),")
    print(
        "with bots drafting purely by ADP + slippage under HARD roster constraints."
    )
    print()

    showcase_team = env.n_teams // 2
    env.reset(our_team_id=showcase_team)
    agent = HierarchicalAgent(position_model, player_models, env)

    with torch.no_grad():
        for rnd in range(1, env.n_rounds + 1):
            env.current_round = rnd
            order = env.get_draft_order(rnd)
            print(f"Round {rnd}:")
            for pick_idx, team_id in enumerate(order, start=1):
                if team_id == showcase_team:
                    context = env.get_context_vector(team_id, rnd)
                    ctx = torch.tensor(
                        context, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    logits = position_model(ctx).squeeze(0)

                    legal_mask = torch.full((4,), float("-inf"), device=device)
                    for i, pos in enumerate(positions):
                        if env.get_available_at_position(team_id, pos):
                            legal_mask[i] = 0.0

                    masked_logits = logits + legal_mask
                    probs = torch.softmax(masked_logits, dim=0).cpu().numpy()
                    qb_p, rb_p, wr_p, te_p = probs

                    needs = env.get_roster_needs(team_id)
                    print(
                        f"  Pick {pick_idx:2d} (our team): "
                        f"QB={qb_p:0.2f}, RB={rb_p:0.2f}, WR={wr_p:0.2f}, TE={te_p:0.2f} | "
                        f"needs: QB={needs['QB']}, RB={needs['RB']}, "
                        f"WR={needs['WR']}, TE={needs['TE']}, FLEX={needs['FLEX']}"
                    )

                    player, _ = agent.pick_player(
                        env, team_id, rnd, epsilon=0.0, training=False
                    )
                    if player is not None:
                        env.make_pick(team_id, player)
                        name = f"{player['first_name']} {player['last_name']}"
                        print(
                            f"    -> Drafted {name} ({player['position']}) "
                            f"ADP={float(player['fantasy_adp']):.1f} "
                            f"PPR={float(player['fantasy_points_ppr']):.1f}"
                        )
                    else:
                        print("    -> No legal pick available for our team!")
                else:
                    bot_player = env.bot_pick(team_id)
                    if bot_player is not None:
                        env.make_pick(team_id, bot_player)

            print()

    print("Final roster for our showcase team:")
    print_agent_roster(env, showcase_team)
    print()
    print("=" * 80)
    print("✓ HIERARCHICAL RL INSPECTION COMPLETE")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run hierarchical training with random seed"""

    import os

    if not os.path.exists("nfl_players_condensed.csv"):
        print("ERROR: Missing nfl_players_condensed.csv")
        return

    # Generate random seed
    random_seed = np.random.randint(0, 1000000)
    
    print("\n" + "=" * 80)
    print("HIERARCHICAL RL - RANDOM SEED VERSION")
    print("=" * 80)
    print(f"Random seed: {random_seed}")
    print("=" * 80)

    # Set all random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    print("\n[STEP 1/3] Training...")
    position_model, player_models, env, scores, ranks = train(
        total_episodes=1500,
        learning_rate=1e-3
    )

    print("\n[STEP 2/3] Plotting training curves...")
    plot_results(scores, ranks)

    print("\n[STEP 3/3] Inspecting learned policy (positions + players)...\n")
    inspect_learned_policy(position_model, player_models, env)


if __name__ == "__main__":
    main()
