python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.01 --model_depth 5 --model_size 512 --norm layer --adaptive_lr True --dataset cifar10 --epochs -1 --name "softhebb wide"

python .\main.py --learning_rule influencehebb --model_type mlp-1 --lr 0.01 --model_depth 5 --model_size 512 --norm layer --adaptive_lr True --dataset cifar10 --epochs -1 --name "simple influence wide softy" --influence_type simple --influencehebb_soft_y True --influencehebb_soft_z False --dot_uw False

python .\main.py --learning_rule influencehebb --model_type mlp-1 --lr 0.01 --model_depth 5 --model_size 512 --norm layer --adaptive_lr True --dataset cifar10 --epochs -1 --name "grad influence wide softy" --influence_type simple --influencehebb_soft_y True --influencehebb_soft_z False --dot_uw False