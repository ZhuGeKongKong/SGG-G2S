#!/usr/bin/env bash
export PYTHONPATH=/home/***/lib/apex:/home/***/lib/cocoapi:/home/***/code/scene_graph_gen/scene_graph_benchmark_pytorch:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=5,6
    export NUM_GUP=2
    echo "Testing Predcls"
    MODEL_NAME="motif_predcls_drop_gate_dist15_2k_FixPModel_CleanHF_FixConfMatDot"
    python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_0014000.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=5,6
    export NUM_GUP=2
    echo "Testing SGCls"
    MODEL_NAME="motif_sgcls_drop_dist15_2k_FixPModel_CleanHF_FixConfMatDot"
    python  -u  -m torch.distributed.launch --master_port 10036 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_0026000.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=5,6
    export NUM_GUP=2
    echo "Testing SGDet"
    MODEL_NAME="motif_sgdet_drop_dist15_2k_FixPModel_CleanHF_FixConfMatDot"
    python  -u  -m torch.distributed.launch --master_port 10038 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_0024000.pth \
            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;

fi
