#!/bin/bash

# classes=('carpet' 'grid' 'leather' 'tile' 'wood')
timestamp=$(date "+%Y%m%d-%H%M%S")
run_id='evalute 1 shot'
run_id2='evalute 2 shot'
shot=1
shot2=2

classes=('carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule' 'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')
for cls in "${classes[@]}"
do
    python main-fewshot.py --train_class "$cls" --run_id "${run_id}" --shot "${shot}"
done 

for cls in "${classes[@]}"
do
    python main-fewshot.py --train_class "$cls" --run_id "${run_id2}" --shot "${shot2}"
done 