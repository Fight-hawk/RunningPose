for file in /data/mxnet/running0813/*
do
    echo $file
    CUDA_VISIBLE_DEVICES=1 python3 single_thread.py --video $file --device GPU --detbatch 1 --outdir ../result  --vis_fast --save_video
done
