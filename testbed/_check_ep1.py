import pandas as pd
df = pd.read_csv('testbed_results/rl_episodes/bid_history.csv', parse_dates=['timestamp'])
df['hour'] = df['timestamp'].dt.hour

ep1 = df[df['episode'] == 1]
ep1_peak = ep1[ep1['hour'].between(7, 17)]
ep1_off = ep1[~ep1['hour'].between(7, 17)]

last_ep = df['episode'].max()
last = df[df['episode'] == last_ep]
last_peak = last[last['hour'].between(7, 17)]

lines = [
    f"Episode 1 peak (H07-H17): mean bid=${ep1_peak['bid_price'].mean():.1f}, std=${ep1_peak['bid_price'].std():.1f}",
    f"Episode 1 off-peak: mean bid=${ep1_off['bid_price'].mean():.1f}, std=${ep1_off['bid_price'].std():.1f}",
    f"Episode 1 all hours: ${ep1['bid_price'].min():.1f} to ${ep1['bid_price'].max():.1f}",
    f"",
    f"Episode {last_ep} peak: mean=${last_peak['bid_price'].mean():.1f}",
    f"Episode {last_ep} all: mean=${last['bid_price'].mean():.1f}",
    f"",
    f"First 5 bids (ep1):",
]
for _, r in ep1.head(5).iterrows():
    lines.append(f"  H{r['hour']:02d}: ${r['bid_price']:.1f} x {r['bid_volume']:.0f}MW")

with open('testbed_results/ep1_check.txt', 'w') as f:
    f.write('\n'.join(lines))
