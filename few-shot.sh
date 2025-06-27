#!/bin/bash

classes=('carpet' 'grid' 'leather' 'tile' 'wood' 'bottle' 'cable' 'capsule' \
'hazelnut' 'metal_nut' 'pill' 'screw' 'toothbrush' 'transistor' 'zipper')

for cls in "${classes[@]}"
do
    python main-fewshot.py --train_class "$cls"
done 