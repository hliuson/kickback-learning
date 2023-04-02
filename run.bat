python .\main.py --learning_rule influencehebb --model_type mlp-1 --lr 0.1 --model_depth 2 --model_size 128 --norm batch --adaptive_lr False --dataset cifar10 --epochs 5 --batch_size 32 --supervised True --name "depth-2"

python .\main.py --learning_rule influencehebb --model_type mlp-1 --lr 0.1 --model_depth 4 --model_size 128 --norm batch --adaptive_lr False --dataset cifar10 --epochs 5 --batch_size 32 --supervised True --name "depth-4"

python .\main.py --learning_rule influencehebb --model_type mlp-1 --lr 0.1 --model_depth 8 --model_size 128 --norm batch --adaptive_lr False --dataset cifar10 --epochs 5 --batch_size 32 --supervised True --name "depth-8"

python .\main.py --learning_rule influencehebb --model_type mlp-1 --lr 0.1 --model_depth 16 --model_size 128 --norm batch --adaptive_lr False --dataset cifar10 --epochs 5 --batch_size 32 --supervised True --name "depth-16"
