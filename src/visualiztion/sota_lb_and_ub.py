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




#plot_sota_lb_and_ub()
plot_ta41_applied_to_ta42_to_ta50()