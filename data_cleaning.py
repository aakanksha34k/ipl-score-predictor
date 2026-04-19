import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading raw dataset...")
# REPLACE 'ball_by_ball_ipl.csv' with your actual filename!
df = pd.read_csv('ball_by_ball_ipl.csv')

print("\n" + "="*50)
print("CHECKPOINT 1: Original Dataset Details")
print("="*50)
print(f"Total Columns ({len(df.columns)}): {list(df.columns)}")
print(f"Dataset Shape: {df.shape}")
print("="*50 + "\n")

print("Mapping your specific dataset columns...")
df = df[df['Innings'].isin([1, 2])]  # Keep only standard innings, ignore super overs
df['batting_team'] = np.where(df['Innings'] == 1, df['Bat First'], df['Bat Second'])
df['bowling_team'] = np.where(df['Innings'] == 1, df['Bat Second'], df['Bat First'])

df.rename(columns={
    'Match ID': 'id',
    'Innings': 'inning',
    'Over': 'over',
    'Ball': 'ball',
    'Venue': 'city',
    'Runs From Ball': 'total_runs',
    'Player Out': 'player_dismissed'
}, inplace=True)

print("\n" + "="*50)
print("CHECKPOINT 2: After Renaming Columns")
print("="*50)
print(f"Mapped Columns: {list(df.columns)}")
print("="*50 + "\n")

print("Filtering for current active IPL teams...")
current_teams = [
    'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
    'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
    'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bengaluru',
    'Sunrisers Hyderabad'
]

df['batting_team'] = df['batting_team'].replace('Delhi Daredevils', 'Delhi Capitals')
df['bowling_team'] = df['bowling_team'].replace('Delhi Daredevils', 'Delhi Capitals')
df['batting_team'] = df['batting_team'].replace('Kings XI Punjab', 'Punjab Kings')
df['bowling_team'] = df['bowling_team'].replace('Kings XI Punjab', 'Punjab Kings')
df['batting_team'] = df['batting_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
df['bowling_team'] = df['bowling_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')

df = df[df['batting_team'].isin(current_teams)]
df = df[df['bowling_team'].isin(current_teams)]

print("Calculating current scores, wickets, and overs...")
df = df.sort_values(['id', 'inning', 'over', 'ball'])

df['current_score'] = df.groupby(['id', 'inning'])['total_runs'].cumsum()

df['player_dismissed'] = df['player_dismissed'].fillna("0")
df['player_dismissed'] = df['player_dismissed'].apply(lambda x: x if x == "0" else "1")
df['player_dismissed'] = df['player_dismissed'].astype('int')
df['wickets_fallen'] = df.groupby(['id', 'inning'])['player_dismissed'].cumsum()
df['wickets_left'] = 10 - df['wickets_fallen']

df['balls_left'] = df['Balls Remaining']
df['balls_bowled'] = 120 - df['balls_left']

print("Calculating Current Run Rate (CRR)...")
df['crr'] = (df['current_score'] * 6) / df['balls_bowled']
df['crr'] = df['crr'].replace([np.inf, -np.inf], 0).fillna(0)

# ── FIX: Calculate 'total' HERE, BEFORE dropping NaN rows from last_five ──
# Previously this was done AFTER dropna, which caused total to be a partial
# sum (runs from ball 30 onwards only) instead of the real final innings score.
# This produced 39,000+ rows where current_score > total — physically impossible.
print("Determining total match scores...")
total_score_df = df.groupby(['id', 'inning'])['total_runs'].sum().reset_index()
total_score_df.rename(columns={'total_runs': 'total'}, inplace=True)
df = df.merge(total_score_df, on=['id', 'inning'])

print("Calculating runs in the last 5 overs...")
groups = df.groupby(['id', 'inning'])
df['last_five'] = groups['total_runs'].rolling(window=30).sum().values
# Drop rows where 30-ball window is not yet filled (first ~5 overs)
df.dropna(subset=['last_five'], inplace=True)

print("Finalizing dataset...")
df.dropna(subset=['city'], inplace=True)

# Drop any remaining impossible rows (data corruption guard)
df = df[df['balls_left'] >= 0]
df = df[df['total'] >= df['current_score']]

final_df = df[[
    'batting_team', 'bowling_team', 'city',
    'current_score', 'balls_left', 'wickets_left',
    'crr', 'last_five', 'total'
]]

final_df = final_df.sample(frac=1)
final_df.to_csv('cleaned_ipl_data.csv', index=False)

print("\n" + "="*50)
print("CHECKPOINT 3: Final Processed Dataset")
print("="*50)
print(f"Final Features ({len(final_df.columns)}): {list(final_df.columns)}")
print(f"Final Dataset Shape: {final_df.shape}")
print("\nHere is a quick peek at the first 3 rows:")
print(final_df.head(3))
print("="*50 + "\n")

# Sanity check: confirm no impossible rows remain
impossible = (final_df['total'] < final_df['current_score']).sum()
print(f"Impossible rows (total < current_score): {impossible}  ← should be 0")
print("Success! 'cleaned_ipl_data.csv' has been generated and is ready for the ML model.")

print("\n" + "="*50)
print("CHECKPOINT 4: Generating Visuals...")
print("="*50)
try:
    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")

    plt.subplot(1, 2, 1)
    sns.histplot(final_df['total'], bins=35, kde=True, color='purple')
    plt.title('Distribution of Total Match Scores', fontsize=14, fontweight='bold')
    plt.xlabel('Total Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.tight_layout()
    plt.savefig('checkpoints_visualized.png', dpi=300)
    print("Visuals generated successfully!")
except Exception as e:
    print(f"Could not generate visuals: {e}")