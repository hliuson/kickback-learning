import wandb
import json
from main import run

def sweep(): 
    #read in sweep config from json
    sweep_config = json.load(open('./sweepcfg/softhebb_sweep.json', 'r'))
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="influencehebb")
    wandb.agent(sweep_id, function=sweeprun, count=256)

def sweeprun():
    wandb.init()
    run(wandb.config)

if __name__ == "__main__":
    sweep()