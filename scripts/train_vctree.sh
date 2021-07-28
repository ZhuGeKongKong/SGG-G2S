#!/usr/bin/env bash
export PYTHONPATH=/home/***/lib/apex:/home/***/lib/cocoapi:/home/***/code/scene_graph_gen/scene_graph_benchmark_pytorch:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=0,2 #3,4 #,4 #3,4
    export NUM_GUP=2
    echo "TRAINING vctree_predcls"
    MODEL_NAME="vctree_predcls_dist15_2k_FixPModel_CleanHF_FixConfMatDot" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10038 --nproc_per_node=$NUM_GUP \
            tools/relation_train_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
            MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
            DTYPE "float32" \
            SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH $NUM_GUP \
            SOLVER.MAX_ITER 50000 \
            SOLVER.VAL_PERIOD 2000 \
            SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.PRETRAINED_MODEL_CKPT ./checkpoints_best/vctree_predcls/model_0030000.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=5,6 #3,4 #,4 #3,4
    export NUM_GUP=2
    echo "TRAINING sgcls"
    MODEL_NAME="vctree_sgcls_drop_dist15_2k_FixPModel_CleanHF_FixConfMatDot" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10040 --nproc_per_node=$NUM_GUP \
          tools/relation_train_net.py \
          --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml" \
          MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
          MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
          MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
          MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
          DTYPE "float32" \
          SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
          SOLVER.MAX_ITER 50000 \
          SOLVER.VAL_PERIOD 2000 \
          SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
          MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
          MODEL.PRETRAINED_MODEL_CKPT ./checkpoints_best/vctree_sgcls_drop/model_0036000.pth \
          OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
          MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
          MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
          MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=2,3 #3,4 #,4 #3,4
    export NUM_GUP=2
    echo "TRAINING sgdet"
    MODEL_NAME="vctree_sgdet_drop_dist15_2k_FixPModel_CleanHF" #"transformer_predcls_dist15_2k_KD0_8_KLt1_freq_TranN2C_1_0_KLt1_InitPreModel_lr1e4"
    mkdir ./checkpoints/${MODEL_NAME}/
    cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/data/datasets/visual_genome.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_transformer.py ./checkpoints/${MODEL_NAME}/
    cp ./maskrcnn_benchmark/layers/gcn/gcn_layers.py ./checkpoints/${MODEL_NAME}/
    python -u -m torch.distributed.launch --master_port 10041 --nproc_per_node=$NUM_GUP \
            tools/relation_train_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_vctree.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor \
            MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
            DTYPE "float32" \
            SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH $NUM_GUP \
            SOLVER.MAX_ITER 50000 \
            SOLVER.VAL_PERIOD 2000 \
            SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.PRETRAINED_MODEL_CKPT ./checkpoints_best/vctree_sgdet_drop/model_0028000.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
fi
