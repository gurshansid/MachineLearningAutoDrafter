"""
Extract 2023 season statistics into a separate CSV file
"""

import pandas as pd

def extract_2023_stats(input_csv='nfl_player_data_with_history.csv', 
                       output_csv='nfl_players_2023_stats.csv'):
    """
    Extract all 2023 season data into a separate CSV
    
    Args:
        input_csv: path to input CSV file with historical data
        output_csv: path to output CSV file for 2023 stats
    """
    
    print(f"Reading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Loaded {len(df)} player-season records")
    
    # Filter to only 2023 season
    df_2023 = df[df['season'] == 2023].copy()
    
    print(f"Found {len(df_2023)} players with 2023 season data")
    
    # Sort by position, then by fantasy points / ADP
    df_2023 = df_2023.sort_values(
        ['fantasy_adp', 'fantasy_points_ppr'], 
        ascending=[True, False]
    )
    
    # Reset index
    df_2023 = df_2023.reset_index(drop=True)
    
    print(f"\nPosition breakdown:")
    print(df_2023['position'].value_counts())
    
    # Show top performers by position
    print("\n=== Top 5 Players by Position (2023 Fantasy Points) ===")
    for pos in ['QB', 'RB', 'WR', 'TE']:
        print(f"\n{pos}:")
        top_pos = df_2023[df_2023['position'] == pos].head(5)[
            ['first_name', 'last_name', 'team', 'fantasy_points_ppr', 'points_per_game']
        ]
        if not top_pos.empty:
            print(top_pos.to_string(index=False))
        else:
            print("  No players found")
    
    # Save to CSV
    df_2023.to_csv(output_csv, index=False)
    print(f"\n✅ 2023 stats saved to '{output_csv}'")
    
    return df_2023


# Example usage
if __name__ == "__main__":
    # Extract 2023 stats
    df_2023 = extract_2023_stats()
    
    # Show sample data
    print("\n=== Sample 2023 Data ===")
    print(df_2023.head(10)[
        ['first_name', 'last_name', 'position', 'team', 'fantasy_points_ppr', 
         'games', 'points_per_game', 'is_rookie']
    ])
    
    # Show rookie stats
    rookies_2023 = df_2023[df_2023['is_rookie'] == True]
    print(f"\n=== 2023 Rookies ({len(rookies_2023)} total) ===")
    print(rookies_2023.head(10)[
        ['first_name', 'last_name', 'position', 'team', 'fantasy_points_ppr', 'nfl_draft_round']
    ])
