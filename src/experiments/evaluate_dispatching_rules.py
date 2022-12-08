import pandas as pd
import matplotlib.pyplot as plt
from src.dispatching_rules.FIFO import FIFO_worker
from src.dispatching_rules.MWKR import MWKR_worker

###############################################################
#                           Globals
###############################################################

EVAL_INSTANCES = [f"taillard/ta{i}.txt" for i in range(41,51)]


def evaluate_instances_on_fifo(instances):
    makespans = {"instance_name": [],"fifo_makespans": []}

    for instance in instances:
        makespan = FIFO_worker(instance)

        makespans["instance_name"].append(instance)
        makespans["fifo_makespans"].append(makespan)

        #print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")

    return makespans

def evaluate_instances_on_mwkr(instances):
    makespans = {"instance_name": [],"mwkr_makespans": []}

    for instance in instances:
        makespan = MWKR_worker(instance)

        makespans["instance_name"].append(instance)
        makespans["mwkr_makespans"].append(makespan)

        #print(f"Mean reward: {mean_reward}\nStd reward: {std_reward}\nMean makespan: {mean_makespan}\nStd makespan: {std_makespan}")

    return makespans

fifo_makespans = evaluate_instances_on_fifo(EVAL_INSTANCES)
mwkr_makespans = evaluate_instances_on_mwkr(EVAL_INSTANCES)


merged_makespans = pd.merge(pd.DataFrame(fifo_makespans), pd.DataFrame(mwkr_makespans), on="instance_name")
merged_makespans.to_csv("logs/sb3_log/evaluate/evaluate_dispatching_rules_on_30x20_instances.csv")

# Replace instance_names with just the instance number and without suffix
merged_makespans['instance_name'] = merged_makespans['instance_name'].str.replace('taillard/','').str.replace('.txt','').str.capitalize()


ax = merged_makespans.set_index("instance_name").plot(kind="bar", figsize=(10,7))
ax.legend(["FIFO", "MWKR"])
ax.set_ylabel("Makespan")
ax.set_xlabel("Instance name")
ax.set_title("Makespan for FIFO and MWKR on Taillard instances with 30 jobs and 20 machines")

fig = ax.get_figure()
plt.xticks(rotation="horizontal")
print(merged_makespans.to_markdown())
fig.savefig("plots/evaluate_dispatching_rules_on_30x20_instances.png", dpi=300)
plt.show()