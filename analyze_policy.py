"""
POLICY ANALYZER - See What Strategy Your Agent Learned

This script shows you:
1. What position the agent drafts in each round (%)
2. Example draft with pick-by-pick explanations
3. Comparison of agent strategy vs random/ADP baseline
4. Policy heatmap visualization

Run after training: python analyze_policy.py
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import from main file
from fantasy_draft_agent import (
    ContextAwarePlayerValueNetwork,
    DraftEnvironment,
    DraftAgent
)


def extract_positional_policy(num_simulations=200):
    """
    Run many drafts and track what positions agent drafts in each round
    
    Returns: DataFrame showing % of time each position is drafted per round
    """
    print("=" * 80)
    print(f"EXTRACTING POSITIONAL POLICY ({num_simulations} simulations)")
    print("=" * 80)
    
    # Load trained model
    df_cond = pd.read_csv("nfl_players_condensed.csv")
    feat_cols = [c for c in df_cond.columns if c not in ["first_name", "last_name"]]
    
    device = torch.device("cpu")
    model = ContextAwarePlayerValueNetwork(
        player_feature_size=len(feat_cols),
        context_size=10
    ).to(device)
    
    model.load_state_dict(torch.load("fantasy_draft_agent.pth", map_location=device))
    model.eval()
    
    # Create environment and agent
    env = DraftEnvironment("nfl_players_2024_stats.csv", n_teams=12, n_rounds=8)
    agent = DraftAgent(model, "nfl_players_condensed.csv")
    
    # Track picks by round
    round_positions = defaultdict(lambda: defaultdict(int))
    
    print(f"\nSimulating {num_simulations} drafts to extract policy...")
    
    for sim in range(num_simulations):
        our_team = env.reset()
        
        for rnd in range(1, 9):
            env.current_round = rnd
            order = env.get_draft_order(rnd)
            
            for tid in order:
                if tid == our_team:
                    # Agent pick (no exploration)
                    player, _, _ = agent.pick_player(
                        env, tid, rnd, epsilon=0.0, training=False
                    )
                    
                    if player is not None:
                        env.make_pick(tid, player)
                        round_positions[rnd][player['position']] += 1
                else:
                    player = env.bot_pick(tid)
                    if player is not None:
                        env.make_pick(tid, player)
        
        if (sim + 1) % 50 == 0:
            print(f"  Completed {sim + 1}/{num_simulations} simulations...")
    
    # Convert to percentages
    policy_data = []
    
    for rnd in range(1, 9):
        counts = round_positions[rnd]
        total = sum(counts.values())
        
        if total > 0:
            policy_data.append({
                'Round': rnd,
                'QB': counts['QB'] / total * 100,
                'RB': counts['RB'] / total * 100,
                'WR': counts['WR'] / total * 100,
                'TE': counts['TE'] / total * 100
            })
    
    df_policy = pd.DataFrame(policy_data)
    
    print("\n✓ Policy extraction complete!")
    
    return df_policy


def display_policy_table(df_policy):
    """Display the learned policy as a table"""
    
    print("\n" + "=" * 80)
    print("LEARNED DRAFT POLICY (% of time each position is drafted)")
    print("=" * 80)
    print("\n" + df_policy.to_string(index=False, float_format=lambda x: f'{x:5.1f}%'))
    
    # Interpretation
    print("\n" + "=" * 80)
    print("POLICY INTERPRETATION")
    print("=" * 80)
    
    for _, row in df_policy.iterrows():
        rnd = int(row['Round'])
        max_pos = max([(row['QB'], 'QB'), (row['RB'], 'RB'), 
                      (row['WR'], 'WR'), (row['TE'], 'TE')])[1]
        max_pct = row[max_pos]
        
        print(f"\nRound {rnd}: Prefers {max_pos} ({max_pct:.1f}% of the time)")
        
        # Add strategic insight
        if rnd <= 3:
            if max_pos in ['RB', 'WR']:
                print(f"  ✓ Good! {max_pos} has high positional scarcity early")
            elif max_pos == 'QB':
                print(f"  ⚠ Unusual - QB usually has less scarcity")
        elif rnd >= 5:
            if max_pos == 'QB':
                print(f"  ✓ Good! Waiting on QB is often optimal")
            elif max_pos in ['RB', 'WR'] and row[max_pos] > 30:
                print(f"  ✓ Still finding value at {max_pos}")


def visualize_policy_heatmap(df_policy):
    """Create a heatmap visualization of the policy"""
    
    print("\n" + "=" * 80)
    print("Creating policy heatmap visualization...")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    heatmap_data = df_policy.set_index('Round')[['QB', 'RB', 'WR', 'TE']]
    
    # Create heatmap
    sns.heatmap(
        heatmap_data.T,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Percentage (%)'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position', fontsize=12, fontweight='bold')
    ax.set_title('Learned Draft Policy: Position Selection by Round', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('policy_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved policy heatmap to policy_heatmap.png")
    plt.close()


def simulate_example_draft_with_explanations():
    """
    Run one detailed draft and explain each pick
    """
    print("\n" + "=" * 80)
    print("EXAMPLE DRAFT WITH PICK-BY-PICK EXPLANATIONS")
    print("=" * 80)
    
    # Load model
    df_cond = pd.read_csv("nfl_players_condensed.csv")
    feat_cols = [c for c in df_cond.columns if c not in ["first_name", "last_name"]]
    
    device = torch.device("cpu")
    model = ContextAwarePlayerValueNetwork(
        player_feature_size=len(feat_cols),
        context_size=10
    ).to(device)
    
    model.load_state_dict(torch.load("fantasy_draft_agent.pth", map_location=device))
    model.eval()
    
    # Environment and agent
    env = DraftEnvironment("nfl_players_2024_stats.csv", n_teams=12, n_rounds=8, seed=42)
    agent = DraftAgent(model, "nfl_players_condensed.csv")
    
    # Run draft from a specific position (middle of draft)
    our_team = env.reset(our_team_id=5)  # 6th pick
    our_picks = []
    
    print(f"\nDrafting from position: {our_team + 1}")
    print("=" * 80)
    
    for rnd in range(1, 9):
        env.current_round = rnd
        order = env.get_draft_order(rnd)
        
        for tid in order:
            if tid == our_team:
                # Get state before pick
                needs = env.get_roster_needs(tid)
                available = env.get_available_players()
                
                print(f"\n{'='*80}")
                print(f"ROUND {rnd} - OUR PICK")
                print(f"{'='*80}")
                print(f"Current Roster: QB={2-needs['QB']}/2  RB={5-env.roster_counts[tid]['RB']}/5  WR={5-env.roster_counts[tid]['WR']}/5  TE={3-env.roster_counts[tid]['TE']}/3")
                print(f"Needs to Fill: QB={needs['QB']}  RB={needs['RB']}  WR={needs['WR']}  TE={needs['TE']}  FLEX={needs['FLEX']}")
                
                # Make pick
                player, _, pick_info = agent.pick_player(
                    env, tid, rnd, epsilon=0.0, training=False
                )
                
                if player is not None:
                    env.make_pick(tid, player)
                    
                    # Explain the pick
                    print(f"\n→ SELECTED: {player['first_name']} {player['last_name']} ({player['position']})")
                    print(f"  ADP: {player['fantasy_adp']:.1f}")
                    print(f"  Projected Points: {player['fantasy_points_ppr']:.1f}")
                    print(f"  VBD: {pick_info['vbd']:.1f}")
                    
                    # Why this pick?
                    print(f"\n  Why this pick?")
                    if needs.get(player['position'], 0) > 0:
                        print(f"    ✓ Fills starter need at {player['position']}")
                    elif needs.get('FLEX', 0) > 0 and player['position'] in ['RB', 'WR', 'TE']:
                        print(f"    ✓ Fills FLEX spot")
                    else:
                        print(f"    • Depth/value pick")
                    
                    if pick_info['vbd'] > 50:
                        print(f"    ✓ High VBD - elite value")
                    elif pick_info['vbd'] > 20:
                        print(f"    • Good VBD - solid value")
                    
                    our_picks.append({
                        'Round': rnd,
                        'Player': f"{player['first_name']} {player['last_name']}",
                        'Pos': player['position'],
                        'ADP': player['fantasy_adp'],
                        'Points': player['fantasy_points_ppr'],
                        'VBD': pick_info['vbd']
                    })
            else:
                # Other teams pick (silent)
                player = env.bot_pick(tid)
                if player is not None:
                    env.make_pick(tid, player)
    
    # Final results
    scores = env.evaluate_league()
    our_score = scores[our_team]
    all_scores = list(scores.values())
    our_rank = sorted(all_scores, reverse=True).index(our_score) + 1
    
    print("\n" + "=" * 80)
    print("DRAFT COMPLETE - OUR TEAM")
    print("=" * 80)
    
    df_picks = pd.DataFrame(our_picks)
    print("\n" + df_picks.to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Total Points: {our_score:.1f}")
    print(f"League Rank:  {our_rank} / 12")
    
    # Show starting lineup
    lineup = env.build_starting_lineup(our_team)
    
    print(f"\n{'='*80}")
    print("STARTING LINEUP")
    print(f"{'='*80}")
    
    for pos in ["QB", "RB", "WR", "TE", "FLEX"]:
        print(f"\n{pos}:")
        for p in lineup[pos]:
            print(f"  {p['first_name']} {p['last_name']}: {p['fantasy_points_ppr']:.1f} pts")
    
    print(f"\nBENCH:")
    for p in lineup["BENCH"]:
        print(f"  {p['first_name']} {p['last_name']} ({p['position']}): {p['fantasy_points_ppr']:.1f} pts")
    
    return our_picks, our_score, our_rank


def compare_policies(num_simulations=100):
    """
    Compare agent's policy to baseline strategies
    """
    print("\n" + "=" * 80)
    print(f"POLICY COMPARISON ({num_simulations} drafts)")
    print("=" * 80)
    
    # Load model
    df_cond = pd.read_csv("nfl_players_condensed.csv")
    feat_cols = [c for c in df_cond.columns if c not in ["first_name", "last_name"]]
    
    device = torch.device("cpu")
    model = ContextAwarePlayerValueNetwork(
        player_feature_size=len(feat_cols),
        context_size=10
    ).to(device)
    
    model.load_state_dict(torch.load("fantasy_draft_agent.pth", map_location=device))
    model.eval()
    
    # Track strategies
    agent_picks = defaultdict(lambda: defaultdict(int))
    baseline_picks = defaultdict(lambda: defaultdict(int))
    
    agent_scores = []
    baseline_scores = []
    
    print("\nTesting RL Agent policy...")
    
    # Test agent
    for sim in range(num_simulations):
        env = DraftEnvironment("nfl_players_2024_stats.csv", n_teams=12, n_rounds=8)
        agent_obj = DraftAgent(model, "nfl_players_condensed.csv")
        
        our_team = env.reset()
        
        for rnd in range(1, 9):
            env.current_round = rnd
            order = env.get_draft_order(rnd)
            
            for tid in order:
                if tid == our_team:
                    player, _, _ = agent_obj.pick_player(env, tid, rnd, 0.0, False)
                    if player is not None:
                        env.make_pick(tid, player)
                        agent_picks[rnd][player['position']] += 1
                else:
                    player = env.bot_pick(tid)
                    if player is not None:
                        env.make_pick(tid, player)
        
        agent_scores.append(env.team_score(our_team))
    
    print("Testing ADP Baseline policy...")
    
    # Test baseline (pure ADP)
    for sim in range(num_simulations):
        env = DraftEnvironment("nfl_players_2024_stats.csv", n_teams=12, n_rounds=8)
        our_team = env.reset()
        
        for rnd in range(1, 9):
            env.current_round = rnd
            order = env.get_draft_order(rnd)
            
            for tid in order:
                player = env.bot_pick(tid)
                if player is not None:
                    env.make_pick(tid, player)
                    if tid == our_team:
                        baseline_picks[rnd][player['position']] += 1
        
        baseline_scores.append(env.team_score(our_team))
    
    # Display comparison
    print("\n" + "=" * 80)
    print("POLICY COMPARISON TABLE")
    print("=" * 80)
    print("\nRL AGENT Policy:")
    print(f"{'Round':<8} {'QB':<10} {'RB':<10} {'WR':<10} {'TE':<10}")
    print("-" * 48)
    
    for rnd in range(1, 9):
        counts = agent_picks[rnd]
        total = sum(counts.values())
        
        qb_pct = counts['QB'] / total * 100 if total > 0 else 0
        rb_pct = counts['RB'] / total * 100 if total > 0 else 0
        wr_pct = counts['WR'] / total * 100 if total > 0 else 0
        te_pct = counts['TE'] / total * 100 if total > 0 else 0
        
        print(f"{rnd:<8} {qb_pct:5.1f}%    {rb_pct:5.1f}%    {wr_pct:5.1f}%    {te_pct:5.1f}%")
    
    print("\n" + "-" * 48)
    print("ADP BASELINE Policy:")
    print(f"{'Round':<8} {'QB':<10} {'RB':<10} {'WR':<10} {'TE':<10}")
    print("-" * 48)
    
    for rnd in range(1, 9):
        counts = baseline_picks[rnd]
        total = sum(counts.values())
        
        qb_pct = counts['QB'] / total * 100 if total > 0 else 0
        rb_pct = counts['RB'] / total * 100 if total > 0 else 0
        wr_pct = counts['WR'] / total * 100 if total > 0 else 0
        te_pct = counts['TE'] / total * 100 if total > 0 else 0
        
        print(f"{rnd:<8} {qb_pct:5.1f}%    {rb_pct:5.1f}%    {wr_pct:5.1f}%    {te_pct:5.1f}%")
    
    # Performance comparison
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    agent_avg = np.mean(agent_scores)
    baseline_avg = np.mean(baseline_scores)
    improvement = agent_avg - baseline_avg
    
    print(f"\nRL Agent:      {agent_avg:.1f} avg points")
    print(f"ADP Baseline:  {baseline_avg:.1f} avg points")
    print(f"Improvement:   {improvement:+.1f} points ({improvement/baseline_avg*100:+.1f}%)")
    
    if improvement > 10:
        print("\n✓ Agent learned a BETTER policy than ADP!")
    elif improvement > 0:
        print("\n~ Agent slightly better than ADP")
    else:
        print("\n✗ Agent not outperforming ADP (needs more training)")


def create_policy_visualization(df_policy):
    """Create comprehensive policy visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Stacked area chart
    ax = axes[0, 0]
    rounds = df_policy['Round'].values
    
    ax.fill_between(rounds, 0, df_policy['QB'], label='QB', alpha=0.7)
    ax.fill_between(rounds, df_policy['QB'], 
                    df_policy['QB'] + df_policy['RB'], label='RB', alpha=0.7)
    ax.fill_between(rounds, df_policy['QB'] + df_policy['RB'],
                    df_policy['QB'] + df_policy['RB'] + df_policy['WR'], 
                    label='WR', alpha=0.7)
    ax.fill_between(rounds, df_policy['QB'] + df_policy['RB'] + df_policy['WR'],
                    100, label='TE', alpha=0.7)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Position Distribution by Round (Stacked)')
    ax.set_xticks(range(1, 9))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Line chart for each position
    ax = axes[0, 1]
    ax.plot(rounds, df_policy['QB'], marker='o', label='QB', linewidth=2)
    ax.plot(rounds, df_policy['RB'], marker='s', label='RB', linewidth=2)
    ax.plot(rounds, df_policy['WR'], marker='^', label='WR', linewidth=2)
    ax.plot(rounds, df_policy['TE'], marker='d', label='TE', linewidth=2)
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Selection Probability (%)')
    ax.set_title('Position Selection Trends by Round')
    ax.set_xticks(range(1, 9))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Heatmap
    ax = axes[1, 0]
    heatmap_data = df_policy.set_index('Round')[['QB', 'RB', 'WR', 'TE']].T
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(1, 9))
    ax.set_yticks(range(4))
    ax.set_yticklabels(['QB', 'RB', 'WR', 'TE'])
    ax.set_xlabel('Round')
    ax.set_ylabel('Position')
    ax.set_title('Policy Heatmap')
    
    # Add text annotations
    for i in range(4):
        for j in range(8):
            text = ax.text(j, i, f'{heatmap_data.iloc[i, j]:.0f}%',
                          ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Percentage (%)')
    
    # 4. Early vs Late round focus
    ax = axes[1, 1]
    
    early_rounds = df_policy[df_policy['Round'] <= 3][['QB', 'RB', 'WR', 'TE']].mean()
    late_rounds = df_policy[df_policy['Round'] >= 6][['QB', 'RB', 'WR', 'TE']].mean()
    
    x = np.arange(4)
    width = 0.35
    
    ax.bar(x - width/2, early_rounds, width, label='Rounds 1-3', alpha=0.8)
    ax.bar(x + width/2, late_rounds, width, label='Rounds 6-8', alpha=0.8)
    
    ax.set_ylabel('Average Selection Rate (%)')
    ax.set_title('Early vs Late Round Position Focus')
    ax.set_xticks(x)
    ax.set_xticklabels(['QB', 'RB', 'WR', 'TE'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('policy_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved detailed policy analysis to policy_analysis.png")
    plt.close()


def show_policy_insights(df_policy):
    """Generate strategic insights from the learned policy"""
    
    print("\n" + "=" * 80)
    print("STRATEGIC INSIGHTS FROM LEARNED POLICY")
    print("=" * 80)
    
    # Early round strategy (1-3)
    early = df_policy[df_policy['Round'] <= 3][['QB', 'RB', 'WR', 'TE']].mean()
    print("\nEarly Rounds (1-3) Focus:")
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pct = early[pos]
        print(f"  {pos}: {pct:5.1f}%", end="")
        if pct > 40:
            print(" ✓ HIGH priority")
        elif pct > 20:
            print(" • Medium priority")
        else:
            print(" ○ Low priority")
    
    # Mid round strategy (4-5)
    mid = df_policy[(df_policy['Round'] >= 4) & (df_policy['Round'] <= 5)][['QB', 'RB', 'WR', 'TE']].mean()
    print("\nMid Rounds (4-5) Focus:")
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pct = mid[pos]
        print(f"  {pos}: {pct:5.1f}%", end="")
        if pct > 40:
            print(" ✓ HIGH priority")
        elif pct > 20:
            print(" • Medium priority")
        else:
            print(" ○ Low priority")
    
    # Late round strategy (6-8)
    late = df_policy[df_policy['Round'] >= 6][['QB', 'RB', 'WR', 'TE']].mean()
    print("\nLate Rounds (6-8) Focus:")
    for pos in ['RB', 'WR', 'QB', 'TE']:
        pct = late[pos]
        print(f"  {pos}: {pct:5.1f}%", end="")
        if pct > 40:
            print(" ✓ HIGH priority")
        elif pct > 20:
            print(" • Medium priority")
        else:
            print(" ○ Low priority")
    
    # Overall strategy assessment
    print("\n" + "=" * 80)
    print("OVERALL STRATEGY ASSESSMENT")
    print("=" * 80)
    
    # Check for good patterns
    rb_early = early['RB']
    wr_early = early['WR']
    qb_early = early['QB']
    qb_mid = mid['QB']
    
    print("\nDoes the agent follow fantasy best practices?")
    
    if rb_early > 30 or wr_early > 30:
        print("  ✓ Prioritizes RB/WR early (scarcity)")
    else:
        print("  ✗ Not prioritizing RB/WR early")
    
    if qb_early < 20:
        print("  ✓ Waits on QB in early rounds")
    else:
        print("  ✗ Drafting QB too early")
    
    if qb_mid > qb_early:
        print("  ✓ QB priority increases in mid rounds (good timing)")
    else:
        print("  • QB timing could be optimized")
    
    # Zero RB strategy check
    if rb_early < 30 and wr_early > 40:
        print("  • Learned 'Zero RB' strategy (WR-heavy early)")
    elif rb_early > 40:
        print("  • Learned 'RB-heavy' strategy (traditional)")
    else:
        print("  • Learned balanced RB/WR approach")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run complete policy analysis"""
    
    import os
    
    # Check for required files
    if not os.path.exists("fantasy_draft_agent.pth"):
        print("\n❌ ERROR: No trained model found!")
        print("Run 'python fantasy_draft_agent.py' first to train the model")
        return
    
    if not os.path.exists("nfl_players_condensed.csv"):
        print("\n❌ ERROR: nfl_players_condensed.csv not found!")
        print("Run 'python fantasy_draft_agent.py' first")
        return
    
    print("\n" + "=" * 80)
    print("FANTASY DRAFT AGENT - POLICY ANALYSIS")
    print("=" * 80)
    
    # 1. Extract policy
    print("\n[ANALYSIS 1/4] Extracting Learned Policy")
    df_policy = extract_positional_policy(num_simulations=200)
    
    # 2. Display policy table
    print("\n[ANALYSIS 2/4] Policy Table")
    display_policy_table(df_policy)
    
    # 3. Show insights
    print("\n[ANALYSIS 3/4] Strategic Insights")
    show_policy_insights(df_policy)
    
    # 4. Create visualizations
    print("\n[ANALYSIS 4/4] Creating Visualizations")
    create_policy_visualization(df_policy)
    visualize_policy_heatmap(df_policy)
    
    # 5. Example draft
    print("\n" + "=" * 80)
    print("[BONUS] Example Draft with Explanations")
    print("=" * 80)
    simulate_example_draft_with_explanations()
    
    # 6. Compare to baseline
    print("\n[COMPARISON] Agent vs Baseline")
    compare_policies(num_simulations=100)
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓ POLICY ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 policy_heatmap.png - Visual representation of policy")
    print("  📈 policy_analysis.png - Detailed policy breakdown")
    print("\nKey Takeaways:")
    print("  • Check the policy table to see round-by-round position preferences")
    print("  • Look for RB/WR emphasis in early rounds (good strategy)")
    print("  • QB should typically be drafted mid-rounds (4-6)")
    print("  • Compare agent vs baseline to see if learning occurred")


if __name__ == "__main__":
    main()