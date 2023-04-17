import wandb
import json
from main import run
import traceback

def sweep(): 
    #read in sweep config from json
    sweep_config = json.load(open('./sweepcfg/simplesofthebb_sweep.json', 'r'))
    print(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="influencehebb")
    wandb.agent(sweep_id, function=sweeprun, count=256)

def sweeprun():
    wandb.init()
    try:
        run(**wandb.config)
    except:
        traceback.print_exc()
        

if __name__ == "__main__":
    sweep()