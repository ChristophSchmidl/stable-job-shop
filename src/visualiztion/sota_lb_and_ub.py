import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Taken from https://optimizizer.com/TA.php

names = [f"Ta{i}" for i in range(41, 51)]
names = names + names
paths = [f"taillard/ta{i}.txt" for i in range(41, 51)]
paths = paths + paths

taillard_instances = {}
taillard_instances["name"] = names
taillard_instances["path"] = paths
lb_values = [1906, 1884, 1809, 1948, 1997, 1957, 1807, 1912, 1931, 1833]
ub_values = [2006, 1939, 1846, 1979, 2000, 2006, 1889, 1937, 1963, 1923]
taillard_instances["lb_ub_values"] = lb_values + ub_values
taillard_instances["Bounds"] = ["Lower bound" for i in lb_values]
taillard_instances["Bounds"].extend(["Upper bound" for i in ub_values])


print(f"Len of names: {len(taillard_instances['name'])}")
print(f"Len of paths: {len(taillard_instances['path'])}")
print(f"Len of lb_ub_values: {len(taillard_instances['lb_ub_values'])}")
print(f"Len of ub_values: {len(taillard_instances['Bounds'])}")


ta_df = pd.DataFrame(taillard_instances)


def plot_sota_lb_and_ub():
    '''
    Plot the lower and upper bounds next to each other as bar plots.
    '''
    sns.barplot(data=ta_df, x='name', y='lb_ub_values', hue='Bounds') 
    plt.xlabel("Instance: 30 jobs, 20 machines")
    plt.ylabel("Makespan")
    plt.title("Lower and upper bounds for Taillard instances")
    plt.show()

def plot_ta41_applied_to_ta42_to_ta50():
    df = pd.read_csv('logs/sb3_log/evaluate/evaluate_model_on_instances.csv')
    df['instance_name'] = df['instance_name'].str.replace('taillard/','')
    sns.barplot(data=df, x='instance_name', y='makespans') 
    plt.xlabel("Instance: 30 jobs, 20 machines")
    plt.ylabel("Makespan")
    plt.title("Ta41 policy applied to Ta41 - Ta50")
    plt.show()

def plot_dispatching_rules_next_to_ta41_applied_to_ta42_to_ta50(instance_name, episodes):
    '''
    Awesome function name :')
    '''
    model_path = f"logs/sb3_log/evaluate/evaluate_model_best_model_{instance_name}_not_tuned_{episodes}_episodes.zip_on_all_instances.csv"


    applied_policy = pd.read_csv(model_path)
    dispatching_rules = pd.read_csv("logs/sb3_log/evaluate/evaluate_dispatching_rules_on_30x20_instances.csv")

    merged_makespans = pd.merge(pd.DataFrame(applied_policy), pd.DataFrame(dispatching_rules), on="instance_name")
    merged_makespans.drop(merged_makespans.filter(regex="Unname"),axis=1, inplace=True)

    #print(merged_makespans)

    '''
           instance_name     rewards  makespans  fifo_makespans  mwkr_makespans
0  taillard/ta41.txt  145.838384     2406.0            2543            2632
1  taillard/ta42.txt   41.232323     2855.0            2578            2401
2  taillard/ta43.txt   65.010101     2636.0            2506            2385
3  taillard/ta44.txt   26.585859     2917.0            2555            2532
4  taillard/ta45.txt   16.686869     2982.0            2565            2431
5  taillard/ta46.txt   21.474747     2976.0            2617            2485
6  taillard/ta47.txt   84.707071     2603.0            2508            2301
7  taillard/ta48.txt   58.020202     2710.0            2541            2350
8  taillard/ta49.txt   38.626263     2784.0            2550            2474
9  taillard/ta50.txt  106.202020     2540.0            2531            2496
    
    '''
    merged_makespans.drop('rewards', axis=1, inplace=True)
    merged_makespans['instance_name'] = merged_makespans['instance_name'].str.replace('taillard/','').str.replace(".txt", '').str.capitalize()

    ax = merged_makespans.set_index("instance_name").plot(kind="bar", figsize=(10,7))
    ax.legend(["RL", "FIFO", "MWKR"])
    ax.set_ylabel("Makespan")
    ax.set_xlabel("Instance name", rotation="horizontal")
    ax.set_title(f"Makespan for RL {instance_name.capitalize()} policy, FIFO and MWKR on Taillard instances with 30 jobs and 20 machines")

    fig = ax.get_figure()
    plt.draw()
    plt.xticks(rotation="horizontal")
    print(merged_makespans.to_markdown())
    fig.savefig(f"plots/compare_dispatching_rules_to_policy_{instance_name}_with_{episodes}_episodes_on_30x20_instances.png", dpi=300)
    plt.show()

#plot_sota_lb_and_ub()
#plot_ta41_applied_to_ta42_to_ta50()
plot_dispatching_rules_next_to_ta41_applied_to_ta42_to_ta50("ta46", "25000")