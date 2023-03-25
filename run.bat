python .\main.py --learning_rule random --model_type cnn-1 --lr 0.01 --model_depth 4 --model_size 32 --norm batch --adaptive_lr False --dataset cifar10 --epochs -1 --name "" --dot_uw False --batch_size 32

python .\main.py --learning_rule end2end --model_type cnn-1 --lr 0.01 --model_depth 4 --model_size 32 --norm batch --adaptive_lr False --dataset cifar10 --epochs -1 --name "" --dot_uw False --batch_size 32

python .\main.py --learning_rule softhebb --model_type cnn-1 --lr 0.01 --model_depth 4 --model_size 32 --norm batch --adaptive_lr True --dataset cifar10 --epochs -1 --name "" --dot_uw False --batch_size 32

python .\main.py --learning_rule influencehebb --model_type cnn-1 --lr 0.01 --model_depth 4 --model_size 32 --norm batch --adaptive_lr True --dataset cifar10 --epochs -1 --name "" --dot_uw False --batch_size 32 --influence_type grad