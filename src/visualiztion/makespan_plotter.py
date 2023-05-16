import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_baseline_comparison(
        csv_file="data/make_spans.csv", 
        title="Makespan comparison with 30 minutes limit",
        value_vars=["Best known makespan", "CP", "FIFO", "MWKR", "RL", "Random"]):
    
    df = pd.read_csv(csv_file, sep=";")

    df_melted = df.melt(id_vars=["Instance", "Jobs-Machines"], value_vars=value_vars, var_name="Method", value_name="Makespan")
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_melted, x="Instance", y="Makespan", hue="Method")
    plt.title(title)
    # Move the legend outside the plot and add padding
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

    # Adjust the right margin to make room for the legend
    plt.subplots_adjust(right=0.8)
    plt.show()



if __name__ == '__main__':
    '''
    plot_baseline_comparison(
        csv_file="./data/make_spans_30_mins_dedicated_policies.csv", 
        title="Makespan comparison: 30 minutes limit, dedicated RL policies",
        value_vars=["Best known makespan", "CP", "FIFO", "MWKR", "RL", "Random"])

    plot_baseline_comparison(
        csv_file="./data/make_spans_30_mins_ta41_applied_to_others.csv", 
        title="Makespan comparison: 30 minutes limit, Ta41 RL policy",
        value_vars=["Best known makespan", "CP", "FIFO", "MWKR", "RL", "Random"])
    
    plot_baseline_comparison(
        csv_file="./data/make_spans_30_mins_merged.csv", 
        title="Makespan comparison: 30 minutes limit",
        value_vars=["Best known makespan", "CP", "FIFO", "MWKR","RL (dedicated policies)","RL (Ta41 policy)", "Random"])
   
    plot_baseline_comparison(
        csv_file="./data/make_spans_rl_compare.csv", 
        title="Makespan comparison: RL",
        value_vars=["Best known makespan", "RL (Dedicated policy, 30mins)", "RL (Ta41 policy, 30mins)", "RL (Ta41 policy, 8 hours)"])
     '''
    plot_baseline_comparison(
        csv_file="./data/make_spans_rl_compare.csv", 
        title="Makespan comparison: RL",
        value_vars=["Best known makespan", "RL (Dedicated policy, 30mins)", "RL (Ta41 policy, 30mins)", "RL (Ta41 policy, 8 hours)", "RL (Ta41 policy, 24 hours)"])