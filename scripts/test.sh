#!/usr/bin/env bash
export PYTHONPATH=/home/***/lib/apex:/home/***/lib/cocoapi:/home/***/code/scene_graph_gen/scene_graph_benchmark_pytorch:$PYTHONPATH
if [ $1 == "0" ]; then
    export CUDA_VISIBLE_DEVICES=2,4
    export NUM_GUP=2
    echo "Testing Predcls"
    MODEL_NAME="transformer_predcls_float32_epoch16_batch16"
    python  -u  -m torch.distributed.launch --master_port 10035 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints_best/${MODEL_NAME}/model_final.pth \
            OUTPUT_DIR ./checkpoints_best/${MODEL_NAME} \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False;

elif [ $1 == "1" ]; then
    export CUDA_VISIBLE_DEVICES=2,3
    export NUM_GUP=2
    echo "Testing Predcls"
    MODEL_NAME="transformer_predcls_dist20_2k_FixPModel_lr1e3_B16_FixCMatDot"
    python  -u  -m torch.distributed.launch --master_port 10036 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints_best/${MODEL_NAME}/model_final.pth \
            OUTPUT_DIR ./checkpoints_best/${MODEL_NAME} \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True  \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
            TEST.ALLOW_LOAD_FROM_CACHE True TEST.VAL_FLAG False;
#    echo "Testing SGCls"
#    MODEL_NAME="transformer_sgdet_dist15_2k_FixPModel_CleanH_Lr1e3_B16"
#    python  -u  -m torch.distributed.launch --master_port 10036 --nproc_per_node=$NUM_GUP \
#            tools/relation_test_net.py \
#            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
#            MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
#            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
#            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerSuperPredictor \
#            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
#            GLOVE_DIR ./datasets/vg/ \
#            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
#            MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_0002000.pth \
#            OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
#            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
#            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0;
elif [ $1 == "2" ]; then
    export CUDA_VISIBLE_DEVICES=2,3
    export NUM_GUP=2
    echo "Testing SGDet"
    MODEL_NAME="transformer_sgdet_dist20_2k_FixPModel_lr1e3_B16_FixCMatDot"
    python  -u  -m torch.distributed.launch --master_port 10036 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints_best/${MODEL_NAME}/model_final.pth \
            OUTPUT_DIR ./checkpoints_best/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER True \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER True;
elif [ $1 == "3" ]; then
    export CUDA_VISIBLE_DEVICES=4
    export NUM_GUP=1
    echo "Testing SGDet"
    MODEL_NAME="transformer_sgdet_Lr1e3_B16_It16"
    python  -u  -m torch.distributed.launch --master_port 10037 --nproc_per_node=$NUM_GUP \
            tools/relation_test_net.py \
            --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_transformer.yaml" \
            MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
            MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
            MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerTransferPredictor \
            TEST.IMS_PER_BATCH $NUM_GUP DTYPE "float32" \
            GLOVE_DIR ./datasets/vg/ \
            MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
            MODEL.WEIGHT ./checkpoints_best/${MODEL_NAME}/model_final.pth \
            OUTPUT_DIR ./checkpoints_best/${MODEL_NAME} \
            TEST.ALLOW_LOAD_FROM_CACHE False TEST.VAL_FLAG False \
            MODEL.ROI_RELATION_HEAD.VAL_ALPHA 0.0 \
            MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER False \
            MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER False;
fi
