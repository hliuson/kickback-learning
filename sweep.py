import wandb
import json
from main import run

def sweep(): 
    #read in sweep config from json
    sweep_config = json.load(open('./sweepcfg/infhebb_sweep.json', 'r'))
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="influencehebb")
    #wandb.agent(sweep_id, function=run, count=100)


if __name__ == "__main__":
    sweep()