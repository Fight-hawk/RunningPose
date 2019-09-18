for file in /data/mxnet/running0917/*
do
    echo $file
    CUDA_VISIBLE_DEVICES=1 python single_thread.py --video $file --device GPU --detbatch 1 --outdir ../result0917  --vis --save_video
done
