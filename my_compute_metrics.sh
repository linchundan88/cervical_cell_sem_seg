#!/bin/bash

# ensemble model
for task_type in "cyto_ins" "nuc_ins"
do
  echo "task_type:${task_type}, model_type: ensemble model"
  python ./my_compute_metrics_ensemble_model.py --task_type ${task_type}
done


 base model
for encoder_weights in "imagenet" "none"
do
  for task_type in "cyto_ins" "nuc_ins"
  do
    for model_type in "Unet_resnet34" "Unet_densenet121" "UnetPlusPlus_resnet34" "UnetPlusPlus_densenet121" "DeepLabV3_resnet34" "DeepLabV3_resnet50" "DeepLabV3Plus_resnet34" "DeepLabV3Plus_resnet50"
    do
      echo "task_type:${task_type}, model_type: ${model_type}, encoder_weights:${encoder_weights}"
      python ./my_compute_metrics_basemodel.py --task_type ${task_type} --model_type ${model_type}  --encoder_weights ${encoder_weights}
    done

  done
done

 transformer based models only have cyto_ins models.
for model_type in "Transunet" "Segformer"
do
  echo "task_type:cyto_ins, model_type: ${model_type}, encoder_weights:none"
  python ./my_compute_metrics_basemodel.py --task_type cyto_ins --model_type  ${model_type}  --encoder_weights none
done
