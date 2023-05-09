#!/bin/bash

PATH_SAVE="/disk_code/code/cervical_cell_sem_seg/trained_models/2023_5_7/"

TRAIN_TIMES=3


for ((i=0;i<TRAIN_TIMES;i++))
do
  for encoder_weights in "imagenet" "none"
  do
    for task_type in "cyto_ins" "nuc_ins"
    do
        if  [ $task_type == "cyto_ins" ] ;
        then
          pos_weight="1"
        fi
        if [ $task_type == "nuc_ins" ] ;
        then
          pos_weight="8"
        fi

#

#        python ./my_train.py --task_type ${task_type} --model_type Unet --encoder_name resnet34  --encoder_weights ${encoder_weights} \
#              --pos_weight ${pos_weight}  --epochs_num 18 --batch_size 64  --save_model_dir ${PATH_SAVE}/${encoder_weights}/Unet_resnet34_times${i}
#        python ./my_train.py --task_type ${task_type} --model_type Unet --encoder_name densenet121  --encoder_weights ${encoder_weights} \
#              --pos_weight ${pos_weight}  --epochs_num 18 --batch_size 64  --save_model_dir ${PATH_SAVE}/${encoder_weights}/Unet_densenet121_times${i}
#        python ./my_train.py --task_type ${task_type} --model_type UnetPlusPlus --encoder_name resnet34  --encoder_weights ${encoder_weights} \
#              --pos_weight ${pos_weight}  --epochs_num 18 --batch_size 64  --save_model_dir ${PATH_SAVE}/${encoder_weights}/UnetPlusPlus_resnet34_times${i}
#        python ./my_train.py --task_type ${task_type} --model_type UnetPlusPlus --encoder_name densenet121  --encoder_weights ${encoder_weights} \
#              --pos_weight ${pos_weight}  --epochs_num 18 --batch_size 32  --save_model_dir ${PATH_SAVE}/${encoder_weights}/UnetPlusPlus_densenet121_times${i}

        python ./my_train.py --task_type ${task_type} --model_type DeepLabV3 --encoder_name resnet34  --encoder_weights ${encoder_weights} \
              --pos_weight ${pos_weight} --epochs_num 18 --batch_size 64  --save_model_dir ${PATH_SAVE}/${encoder_weights}/DeepLabV3_resnet34_times${i}
        python ./my_train.py --task_type ${task_type} --model_type DeepLabV3 --encoder_name resnet50  --encoder_weights ${encoder_weights} \
              --pos_weight ${pos_weight} --epochs_num 18 --batch_size 64  --save_model_dir ${PATH_SAVE}/${encoder_weights}/DeepLabV3_resnet50_times${i}
        python ./my_train.py --task_type ${task_type} --model_type DeepLabV3Plus --encoder_name resnet34  --encoder_weights ${encoder_weights} \
              --pos_weight ${pos_weight} --epochs_num 18 --batch_size 64  --save_model_dir ${PATH_SAVE}/${encoder_weights}/DeepLabV3Plus_resnet34_times${i}
        python ./my_train.py --task_type ${task_type} --model_type DeepLabV3Plus --encoder_name resnet50  --encoder_weights ${encoder_weights} \
              --pos_weight ${pos_weight} --epochs_num 18 --batch_size 64  --save_model_dir ${PATH_SAVE}/${encoder_weights}/DeepLabV3Plus_resnet50_times${i}

      done
  done

done



for ((i=0;i<TRAIN_TIMES;i++))
do
  for task_type in "cyto_ins" "nuc_ins"
  do
      if  [ $task_type == "cyto_ins" ] ;
      then
        pos_weight="1"
      fi
      if [ $task_type == "nuc_ins" ] ;
      then
        pos_weight="8"
      fi

      python ./my_train.py --task_type ${task_type} --model_type Segformer --encoder_name none  --encoder_weights none \
            --pos_weight ${pos_weight} --epochs_num 18 --batch_size 32  --save_model_dir ${PATH_SAVE}/${encoder_weights}/DeepLabV3_resnet34_times${i}
      python ./my_train.py --task_type ${task_type} --model_type Transunet --encoder_name none  --encoder_weights none \
            --pos_weight ${pos_weight} --epochs_num 18 --batch_size 32  --save_model_dir ${PATH_SAVE}/${encoder_weights}/DeepLabV3_resnet50_times${i}

    done


done
