from IPython.display import display
import pandas as pd

df = pd.read_csv('logs/sb3_log/progress.csv')

#print(df.to_string()) 
display(df)