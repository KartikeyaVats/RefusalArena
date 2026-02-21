import pandas as pd

def readAdvBench():
  url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
  advbench_df = pd.read_csv(url)
  prompts = advbench_df["goal"].astype(str).tolist()
  targets = advbench_df["target"].astype(str).tolist()
  return prompts, targets
