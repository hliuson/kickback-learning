python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 32 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-32" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule simplesofthebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 32 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-32" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule hebbnet --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 32 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-32" --dropout True --probe True --activation relu --supervised False   

python .\main.py --learning_rule random --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 32 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-32" --dropout True --probe True --activation relu --supervised False   

python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-128" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule simplesofthebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-128" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule hebbnet --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-128" --dropout True --probe True --activation relu --supervised False   

python .\main.py --learning_rule random --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-128" --dropout True --probe True --activation relu --supervised False   

python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 512 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-512" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule simplesofthebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 512 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-512" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule hebbnet --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 512 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-512" --dropout True --probe True --activation relu --supervised False   

python .\main.py --learning_rule random --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 512 --norm layer --dataset mnist --epochs 1 --batch_size 32 --name "Skew-512" --dropout True --probe True --activation relu --supervised False   

python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset cifar10 --epochs 1 --batch_size 32 --name "Skew-128-cifar" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule simplesofthebb --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset cifar10 --epochs 1 --batch_size 32 --name "Skew-128-cifar" --dropout True --probe True --activation relu --supervised False

python .\main.py --learning_rule hebbnet --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset cifar10 --epochs 1 --batch_size 32 --name "Skew-128-cifar" --dropout True --probe True --activation relu --supervised False   

python .\main.py --learning_rule random --model_type mlp-1 --lr 0.05 --model_depth 2 --model_size 128 --norm layer --dataset cifar10 --epochs 1 --batch_size 32 --name "Skew-128-cifar" --dropout True --probe True --activation relu --supervised False   