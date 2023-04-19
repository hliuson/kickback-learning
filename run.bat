python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.01 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 3 --batch_size 32 --name "DEAD3" --temp 0.25 --dropout True --probe True --activation triangle --group_size -1 --shuffle False

python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.01 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 3 --batch_size 32 --name "DEAD3" --temp 1 --dropout True --probe True --activation triangle --group_size -1 --shuffle False

python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.01 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 3 --batch_size 32 --name "DEAD3" --temp 4 --dropout True --probe True --activation triangle --group_size -1 --shuffle False

python .\main.py --learning_rule softhebb --model_type mlp-1 --lr 0.01 --model_depth 2 --model_size 128 --norm layer --dataset mnist --epochs 3 --batch_size 32 --name "DEAD3" --temp 16 --dropout True --probe True --activation triangle --group_size -1 --shuffle False