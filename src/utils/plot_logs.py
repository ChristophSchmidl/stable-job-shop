import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd


log_df = pd.read_csv('logs/sb3_log/reward_log.csv')

sns.set_style('darkgrid')

fig, ax1 = plt.subplots()
line_labels = ["Reward", "Makespan"]

color1 = 'blue'
line1 = sns.lineplot(x = "episode", y = "reward", color=color1, data=log_df)
ax1.set_ylabel('Reward', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()

color2 = 'orange'
line2 = sns.lineplot(x = "episode", y= "makespan", color=color2, ax=ax2, data=log_df)
ax2.set_ylabel('Makespan', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.grid(False) # no grid over top of plots

# see: https://stackoverflow.com/questions/39500265/how-to-manually-create-a-legend

reward_line = mlines.Line2D([], [], color=color1, label='Reward')
makespan_line = mlines.Line2D([], [], color=color2, label='Makespan')

# FIFO - for ta41
ax2.axhline(y=2543, color='green', marker='o', linestyle='--', linewidth = 4);
# MWKR - for ta41
ax2.axhline(y=2632, color='red', marker='o', linestyle='--', linewidth = 4);

fifo_line = mlines.Line2D([], [], color="green", label='FIFO (2543)')
mwkr_line = mlines.Line2D([], [], color="red", label='MWKR (2632)')

#reward_patch = mpatches.Patch(color=color1, label='Reward')
#makespan_patch = mpatches.Patch(color=color2, label='Makespam')

plt.legend(handles=[reward_line, makespan_line, fifo_line, mwkr_line], loc='lower left')

#ax2.legend(loc="upper right");
#plt.legend(labels=labels, loc='upper right')

plt.show()