"""
TRULY PURE RL - Zero Domain Knowledge

Restart criteria: ONLY based on performance (score improvement)
No assumptions about QB/RB/WR strategy at all

Agent must discover everything from scratch through experience
"""

import copy
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time


# ============================================================================
# NETWORK
# ============================================================================


class ContextAwarePlayerValueNetwork(nn.Module):
    def __init__(self, player_feature_size: int, context_size: int = 10):
        super().__init__()

        # Encodes the draft context (round, pick, roster needs, etc.)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Encodes the "player" feature vector – in our case, a 4-dim
        # one-hot for position (QB/RB/WR/TE), but this stays generic.
        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Combine context + player/position into a single scalar "value"
        self.decision_head = nn.Sequential(
            nn.Linear(16 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, player_features: torch.Tensor, context_features: torch.Tensor) -> torch.Tensor:
        """
        player_features: [batch, player_feature_size]
        context_features: [batch, context_size]
        returns: [batch, 1] scores for each (context, player/position) pair
        """
        context_encoded = self.context_encoder(context_features)
        player_encoded = self.player_encoder(player_features)
        combined = torch.cat([context_encoded, player_encoded], dim=1)
        return self.decision_head(combined)


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

        # Load condensed CSV for drafting (with ADP)
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
        # Match on first_name and last_name
        df_draft = df_draft.merge(
            df_scoring[["first_name", "last_name", "fantasy_points_ppr"]],
            on=["first_name", "last_name"],
            how="left",
            suffixes=("", "_actual"),
        )

        # Use actual 2024 fantasy_points_ppr for scoring
        # If no match found, use a default low score
        df_draft["fantasy_points_ppr"] = df_draft["fantasy_points_ppr"].fillna(0.0)

        self.player_pool = df_draft
        self.reset()
        print(f"[Environment] Loaded {len(df_draft)} players for drafting")
        print(f"[Environment] Using actual 2024 stats for scoring")

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

    def position_available_for_team(self, team_id: int, position: str) -> bool:
        """
        Check whether this team can draft the given position *and*
        there is at least one remaining player at that position.
        """
        if not self.can_draft_position(team_id, position):
            return False

        available = self.get_available_players()
        if available.empty:
            return False

        return (available["position"] == position).any()

    def pick_best_player_at_position(self, team_id: int, position: str):
        """
        Return the best remaining ADP player at the given position
        that this team is allowed to draft. If none are available,
        return None.
        """
        available = self.get_available_players()
        if available.empty:
            return None

        if not self.can_draft_position(team_id, position):
            return None

        avail_pos = available[available["position"] == position]
        if avail_pos.empty:
            return None

        # player_pool is ADP-sorted, and available preserves that order,
        # so the first row is the best ADP at this position.
        return avail_pos.iloc[0]

    def evaluate_league(self):
        return {tid: self.team_score(tid) for tid in range(self.n_teams)}


# ============================================================================
# AGENT (POSITION POLICY)
# ============================================================================


class DraftAgent:
    """
    Position-policy agent.

    The action is *which position to draft* (QB/RB/WR/TE).
    After the agent chooses a position, the environment selects
    the best remaining ADP player at that position.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

        # Fixed discrete action space
        self.positions = ["QB", "RB", "WR", "TE"]
        self.pos_to_idx = {p: i for i, p in enumerate(self.positions)}

    def get_position_features(self, position: str):
        """Return a 4-dim one-hot vector encoding the position."""
        vec = np.zeros(len(self.positions), dtype=np.float32)
        vec[self.pos_to_idx[position]] = 1.0
        return vec

    def pick_position(self, env, team_id: int, round_num: int, epsilon: float, training: bool = True):
        """
        Sample a position to draft for this pick.

        - Build a candidate set of legal positions (roster limit + players left).
        - For each candidate position, form (position_one_hot, context_vector).
        - Use the model to score each candidate.
        - ε-greedy: with probability ε pick a random legal position,
          otherwise pick argmax over the model scores.
        """
        context = env.get_context_vector(team_id, round_num)
        context = np.asarray(context, dtype=np.float32)

        legal_positions = []
        pos_feats = []

        for pos in self.positions:
            if env.position_available_for_team(team_id, pos):
                legal_positions.append(pos)
                pos_feats.append(self.get_position_features(pos))

        if not legal_positions:
            return None, None

        pos_tensor = torch.tensor(
            np.stack(pos_feats),
            dtype=torch.float32,
            device=self.device,
        )

        context_batch = torch.tensor(
            np.repeat(context[None, :], len(legal_positions), axis=0),
            dtype=torch.float32,
            device=self.device,
        )

        if training:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(training):
            logits = self.model(pos_tensor, context_batch).squeeze(1)
            probs = torch.softmax(logits, dim=0)

            if training and random.random() < epsilon:
                idx = random.randrange(len(legal_positions))
            else:
                idx = int(torch.argmax(probs).item())

            log_prob = torch.log(probs[idx] + 1e-10) if training else None

        chosen_pos = legal_positions[idx]
        return chosen_pos, log_prob


# ============================================================================
# PURE PERFORMANCE-BASED RESTART DETECTION
# ============================================================================


def should_restart_based_on_performance(scores, score_threshold=1800, min_improvement=5):
    """
    Heuristic for deciding when to restart training:

    - If fewer than 400 episodes: never restart (too early)
    - Otherwise:
        • Look at the last 200 vs previous 200 episodes
        • If recent avg >= threshold → NEVER restart (we're good)
        • Else if improvement < min_improvement → restart
    """
    if len(scores) < 400:
        return False, None

    recent_200 = scores[-200:]
    previous_200 = scores[-400:-200]

    recent_avg = np.mean(recent_200)
    previous_avg = np.mean(previous_200)
    improvement = recent_avg - previous_avg

    if recent_avg >= score_threshold:
        return False, None

    if improvement < min_improvement:
        return True, {
            "recent_avg": recent_avg,
            "previous_avg": previous_avg,
            "improvement": improvement,
        }

    return False, None


# ============================================================================
# TRAINING WITH PERFORMANCE-ONLY RESTARTS
# ============================================================================


def train_pure_performance_restarts(
    total_episodes=5000,
    restart_check_window=300,
    score_threshold=1800,
    max_restarts=5,
    learning_rate=1e-3,
):
    """
    Train with restarts based ONLY on score threshold
    """

    print("=" * 80)
    print("TRULY PURE RL - Score Threshold Restarts")
    print("=" * 80)
    print("\nPhilosophy:")
    print("  • NO domain knowledge about positions")
    print("  • NO hints about when to draft QB/RB/WR")
    print("  • Restart based ONLY on performance")
    print("\nBot opponents:")
    print("  • Draft from top ADP players (with slippage)")
    print("  • Makes them beatable - agent has room to learn")
    print("\nRestart trigger (checked every 200 episodes):")
    print(f"  • Score < {score_threshold} AND improving < 5 points per 200 eps → RESTART")
    print("  • Score >= threshold → NEVER restart (optimal!)")
    print("=" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We now treat "player features" as a 4-dim one-hot vector for position:
    # [QB, RB, WR, TE]
    player_feature_size = 4

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

    all_scores = []
    all_ranks = []
    all_rewards = []
    restart_episodes = []

    best_model_state = None
    best_avg_score = -float("inf")

    num_restarts = 0
    episodes_done = 0
    current_attempt = 1

    print(f"\nATTEMPT {current_attempt}/{max_restarts + 1}\n")

    model = ContextAwarePlayerValueNetwork(
        player_feature_size=player_feature_size,
        context_size=10,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    agent = DraftAgent(model)

    while episodes_done < total_episodes:
        ep = episodes_done

        epsilon = max(0.05, 0.9 - 0.85 * (ep / total_episodes))

        our_team = env.reset()
        log_probs = []

        for rnd in range(1, 9):
            env.current_round = rnd
            order = env.get_draft_order(rnd)

            for tid in order:
                if tid == our_team:
                    position, log_prob = agent.pick_position(env, tid, rnd, epsilon, training=True)
                    if position is not None:
                        player = env.pick_best_player_at_position(tid, position)
                        if player is None:
                            player = env.bot_pick(tid)
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

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        all_scores.append(our_score)
        all_ranks.append(our_rank)
        all_rewards.append(reward)

        episodes_done += 1

        if episodes_done % 50 == 0:
            window = min(100, len(all_scores))
            avg_score = np.mean(all_scores[-window:])
            avg_rank = np.mean(all_ranks[-window:])

            print(
                f"Ep {episodes_done:4d}/{total_episodes} | "
                f"Score:{our_score:6.1f} Rank:{our_rank:2d}/12 | "
                f"Avg{window}[S:{avg_score:6.1f} Rk:{avg_rank:4.1f}] | "
                f"Attempt:{current_attempt} | ε:{epsilon:.3f}"
            )

        if len(all_scores) >= 100:
            recent_avg = np.mean(all_scores[-100:])
            if recent_avg > best_avg_score:
                best_avg_score = recent_avg
                best_model_state = copy.deepcopy(model.state_dict())

        if episodes_done % 200 == 0 and episodes_done >= 400:
            should_restart, reason = should_restart_based_on_performance(
                all_scores,
                score_threshold=score_threshold,
            )

            if should_restart and num_restarts < max_restarts:
                recent_avg = np.mean(all_scores[-200:])
                prev_avg = np.mean(all_scores[-400:-200])
                print("\n" + "-" * 60)
                print(f"RESTARTING at episode {episodes_done}!")
                print(f"  Previous 200 avg: {prev_avg:.1f}")
                print(f"  Recent   200 avg: {recent_avg:.1f}")
                print(
                    f"  Improvement:      {recent_avg - prev_avg:.1f} "
                    f"(threshold {reason['improvement']:.1f})"
                )
                print("-" * 60 + "\n")

                restart_episodes.append(episodes_done)
                num_restarts += 1
                current_attempt += 1

                model = ContextAwarePlayerValueNetwork(
                    player_feature_size=player_feature_size,
                    context_size=10,
                ).to(device)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                agent = DraftAgent(model)

            elif not should_restart:
                recent_avg = np.mean(all_scores[-200:])
                print(
                    f"\nNo restart at ep {episodes_done}: "
                    f"recent 200 avg score = {recent_avg:.1f} "
                    f"(threshold {score_threshold}) - continuing...\n"
                )

    print("\nTraining complete.")
    print(f"Total restarts: {num_restarts}")
    if restart_episodes:
        print("Restart episodes: ", restart_episodes)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "fantasy_draft_agent.pth")
        print("\n✓ Saved best-performing model to fantasy_draft_agent.pth")

    return model, all_scores, all_ranks, all_rewards, restart_episodes


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_training(scores, ranks, rewards, restart_points):
    """Plot with restart points marked"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scores
    ax = axes[0, 0]
    ax.plot(scores, alpha=0.3)
    if len(scores) >= 100:
        ma = pd.Series(scores).rolling(100).mean()
        ax.plot(ma, linewidth=2.5, label="100-Ep MA")

    for i, restart_ep in enumerate(restart_points):
        ax.axvline(x=restart_ep, linestyle="--", linewidth=2, label="Restart" if i == 0 else "")

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
    """Pure RL with performance-based restarts"""

    print("\n" + "=" * 80)
    print("PURE RL - Performance-Based Restarts")
    print("=" * 80)

    print("\n[STEP 1/2] Training")
    print("This will take a bit for 5000 episodes...\n")

    model, scores, ranks, rewards, restarts = train_pure_performance_restarts(
        total_episodes=5000,
        restart_check_window=300,
        score_threshold=1800,
        max_restarts=5,
        learning_rate=1e-3,
    )

    print("\n[STEP 2/2] Plotting training curves...")
    plot_training(scores, ranks, rewards, restarts)

    print("\n" + "=" * 80)
    print("✓ PURE RL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nRestarts: {len(restarts)}")
    if restarts:
        print(f"Restart episodes: {restarts}")
    print("\nNext: python analyze_policy.py")


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
