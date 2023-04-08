python .\main.py --learning_rule influencehebb --model_type mlp-1 --lr 0.1 --model_depth 2 --model_size 256 --norm batch --adaptive_lr True --dataset cifar10 --epochs 5 --batch_size 32 --supervised True --name "" --influence_type grad

python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.1 --model_depth 2 --model_size 256 --norm batch --adaptive_lr True --dataset cifar10 --epochs 5 --batch_size 32 --supervised True --name "" --influence_type grad
