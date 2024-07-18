for i in {0..0}
do
    WORLD_SIZE=1 CUDA_VISIBLE_DEVICES=3 torchrun \
                --nproc_per_node=1 \
                --nnodes=1          \
                --node_rank=0       \
                --master_addr=localhost  \
                --master_port=22224 \
                train.py "train.random_seed=$i"
done