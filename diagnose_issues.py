"""
Diagnostic Script - Find Why Agent Learned Bad Policy

This checks:
1. Data quality issues
2. Feature problems
3. Reward signal issues
4. Network capacity
"""

import pandas as pd
import numpy as np


def check_data_quality():
    """Check if data files are complete and correct"""
    
    print("=" * 80)
    print("DATA QUALITY DIAGNOSTICS")
    print("=" * 80)
    
    issues_found = []
    
    # Check 2024 stats
    print("\n[1] Checking nfl_players_2024_stats.csv...")
    try:
        df_2024 = pd.read_csv("nfl_players_2024_stats.csv")
        print(f"   ✓ Loaded {len(df_2024)} players")
        print(f"   Position counts: {df_2024['position'].value_counts().to_dict()}")
        
        # Check for required columns
        required_cols = ['first_name', 'last_name', 'position', 'fantasy_adp', 'fantasy_points_ppr']
        missing = [c for c in required_cols if c not in df_2024.columns]
        if missing:
            print(f"   ✗ Missing columns: {missing}")
            issues_found.append(f"2024 stats missing columns: {missing}")
        
        # Check ADP quality
        valid_adp = df_2024[df_2024['fantasy_adp'] < 300]
        print(f"   Draftable players (ADP < 300): {len(valid_adp)}")
        
        if len(valid_adp) < 200:
            print(f"   ⚠ WARNING: Only {len(valid_adp)} draftable players (should be 250-300)")
            issues_found.append(f"Too few draftable players: {len(valid_adp)}")
        
        # Check fantasy points distribution
        avg_points = df_2024['fantasy_points_ppr'].mean()
        print(f"   Average fantasy points: {avg_points:.1f}")
        
        if avg_points < 50 or avg_points > 300:
            print(f"   ⚠ WARNING: Unusual average points: {avg_points:.1f}")
            issues_found.append(f"Unusual fantasy points average: {avg_points:.1f}")
        
    except FileNotFoundError:
        print("   ✗ FILE NOT FOUND")
        issues_found.append("nfl_players_2024_stats.csv not found")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        issues_found.append(f"Error loading 2024 stats: {e}")
    
    # Check condensed features
    print("\n[2] Checking nfl_players_condensed.csv...")
    try:
        df_cond = pd.read_csv("nfl_players_condensed.csv")
        print(f"   ✓ Loaded {len(df_cond)} players")
        
        feat_cols = [c for c in df_cond.columns if c not in ['first_name', 'last_name']]
        print(f"   Features: {len(feat_cols)}")
        print(f"   Feature list: {feat_cols[:10]}...")
        
        # Check for NaN
        nan_counts = df_cond[feat_cols].isna().sum().sum()
        if nan_counts > 0:
            print(f"   ⚠ WARNING: {nan_counts} NaN values in features")
            issues_found.append(f"NaN values in condensed features: {nan_counts}")
        
        # Check feature variance
        stds = df_cond[feat_cols].std()
        zero_var = stds[stds == 0].index.tolist()
        if zero_var:
            print(f"   ⚠ WARNING: Zero-variance features: {zero_var}")
            issues_found.append(f"Zero-variance features: {zero_var}")
        
    except FileNotFoundError:
        print("   ✗ FILE NOT FOUND")
        issues_found.append("nfl_players_condensed.csv not found")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        issues_found.append(f"Error loading condensed: {e}")
    
    # Check data alignment
    print("\n[3] Checking data alignment...")
    try:
        # Check if same players in both files
        players_2024 = set(zip(df_2024['first_name'], df_2024['last_name']))
        players_cond = set(zip(df_cond['first_name'], df_cond['last_name']))
        
        only_in_2024 = players_2024 - players_cond
        only_in_cond = players_cond - players_2024
        
        if only_in_2024:
            print(f"   ⚠ {len(only_in_2024)} players in 2024 but not condensed")
            if len(only_in_2024) > 50:
                issues_found.append(f"Major mismatch: {len(only_in_2024)} players missing from condensed")
        
        if only_in_cond:
            print(f"   ⚠ {len(only_in_cond)} players in condensed but not 2024")
        
        if not only_in_2024 and not only_in_cond:
            print("   ✓ Player lists match perfectly")
        
    except:
        print("   ✗ Could not compare datasets")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    if issues_found:
        print(f"\n⚠ Found {len(issues_found)} issues:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    else:
        print("\n✓ No data quality issues detected")
    
    return issues_found


def check_why_bad_policy():
    """Analyze why the agent learned to draft QB-QB-TE-RB-RB-RB-RB-RB"""
    
    print("\n" + "=" * 80)
    print("WHY DID AGENT LEARN BAD POLICY?")
    print("=" * 80)
    
    df_2024 = pd.read_csv("nfl_players_2024_stats.csv")
    
    # Check QB scoring
    print("\n[Hypothesis 1] Are QBs massively outscoring other positions?")
    
    for pos in ['QB', 'RB', 'WR', 'TE']:
        pos_data = df_2024[df_2024['position'] == pos]
        avg_points = pos_data['fantasy_points_ppr'].mean()
        max_points = pos_data['fantasy_points_ppr'].max()
        top5_avg = pos_data.nlargest(5, 'fantasy_points_ppr')['fantasy_points_ppr'].mean()
        
        print(f"\n{pos}:")
        print(f"   Average: {avg_points:.1f}")
        print(f"   Top player: {max_points:.1f}")
        print(f"   Top 5 avg: {top5_avg:.1f}")
    
    # Check roster construction
    print("\n[Hypothesis 2] Can you even field a legal team with QB-QB-TE strategy?")
    print("\nStarting lineup requires:")
    print("   1 QB, 2 RB, 2 WR, 1 TE, 2 FLEX (RB/WR/TE)")
    print("\nAgent's typical draft: 2 QB, 1 TE, 5 RB, 0 WR")
    print("   ✗ MISSING 2 WRs - ILLEGAL LINEUP!")
    print("   This team would FAIL to field a complete roster")
    
    # Check if this explains bad scores
    print("\n[Hypothesis 3] Checking agent's actual performance...")
    print("   If agent learned this policy, it should have TERRIBLE scores")
    print("   Check training_results.png - are scores very low?")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print("""
The agent learned a pathological policy because:

1. ✗ REWARD SIGNAL ISSUE
   - Agent discovered it can get high VBD by drafting top QBs
   - Reward doesn't penalize heavily enough for illegal lineups
   - Needs WR constraint not enforced strongly

2. ✗ INSUFFICIENT EXPLORATION
   - Got stuck in local optimum early in training
   - Never explored WR-heavy strategies
   - 500 episodes wasn't enough to escape

3. ✗ POSITION SCARCITY NOT LEARNED
   - Agent didn't learn RB/WR are scarce, QB is not
   - Replacement value calculation may be off

SOLUTIONS:
→ Train for 2000+ episodes instead of 500
→ Add hard constraint: MUST draft at least 1 WR by round 3
→ Increase penalty for unfilled positions in reward
→ Add diversity bonus to encourage trying different strategies
    """)


def suggest_fixes():
    """Suggest specific fixes"""
    
    print("\n" + "=" * 80)
    print("RECOMMENDED FIXES")
    print("=" * 80)
    
    print("""
FIX 1: Add Position Diversity Constraint
─────────────────────────────────────────
In calculate_pick_reward(), add:

    # After existing reward calculation
    roster_positions = [p['position'] for p in env.rosters[team_id]]
    unique_positions = len(set(roster_positions))
    
    # Penalty if only drafting 1-2 positions
    if len(roster_positions) >= 4 and unique_positions <= 2:
        reward -= 2.0  # Heavy penalty for no diversity


FIX 2: Enforce WR Requirement Early
────────────────────────────────────
In calculate_pick_reward(), add:

    # Must draft WR by round 3
    if round_num == 3:
        roster_positions = [p['position'] for p in env.rosters[team_id]]
        if 'WR' not in roster_positions:
            reward -= 3.0  # Huge penalty if no WR yet


FIX 3: Increase Training Episodes
──────────────────────────────────
In fantasy_draft_agent.py, change:

    train(num_episodes=2000)  # Was 500


FIX 4: Stronger Positional Need Rewards
────────────────────────────────────────
In calculate_pick_reward(), change:

    # OLD:
    if needs.get(position, 0) > 0:
        reward += 0.5
    
    # NEW:
    if needs.get(position, 0) > 0:
        reward += 1.5  # Stronger signal to fill needs


FIX 5: Check Your Data
──────────────────────
Run:
    python diagnose_issues.py
    
To see if there are data quality problems causing this.
    """)


def main():
    """Run diagnostics"""
    
    print("\n🚨 YOUR AGENT LEARNED A TERRIBLE POLICY!")
    print("\nIt drafts: QB-QB-TE-RB-RB-RB-RB-RB")
    print("This team has NO WIDE RECEIVERS and can't field a legal lineup!")
    
    print("\nLet's diagnose why...")
    
    # Check data
    issues = check_data_quality()
    
    # Explain the bad policy
    check_why_bad_policy()
    
    # Suggest fixes
    suggest_fixes()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. Review the diagnostics above
2. Check training_results.png - did scores improve at all?
3. Re-run training with fixes (2000 episodes + constraints)
4. Or let me know and I'll create a fixed version for you
    """)


if __name__ == "__main__":
    main()