CUDA_VISIBLE_DEVICES=0 \
python main.py \
--mode testAUC \
--image_dir datasets/covid \
--dataset Covid \
--image_size 256 \
--sample_dir covid/samples \
--log_dir covid/logs \
--model_save_dir covid/models \
--result_dir covid/result \
--test_iters 100 \
--batch_size 1 \
--num_workers 2
