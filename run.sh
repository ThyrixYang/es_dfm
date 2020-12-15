# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
# --method Pretrain --mode pretrain --model_ckpt_path /home/yangjq/Working/delayed_feedback_release/ckpts/pretrain/baseline \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
# --method FSIW --mode pretrain --model_ckpt_path /home/yangjq/Working/delayed_feedback_release/ckpts/pretrain/fsiw0 \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache \
# --fsiw_pretraining_type fsiw1

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
# --method DFM --mode pretrain --model_ckpt_path /home/yangjq/Working/delayed_feedback_release/ckpts/pretrain/dfm \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache \
# --epoch 50

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
# --method ES-DFM --mode pretrain --model_ckpt_path /home/yangjq/Working/delayed_feedback_release/ckpts/pretrain/esdfm \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
# --method FNW --mode stream \
# --pretrain_baseline_model_ckpt_path /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss.tf \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
# --method ES-DFM --mode stream \
# --pretrain_baseline_model_ckpt_path /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss.tf \
# --pretrain_esdfm_model_ckpt_path /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_tn_dp_first_30_train_last_30_test_tn_dp_prtrain_delay_tn_dp_loss/MLP_tn_dp_first_30_train_last_30_test_tn_dp_prtrain_delay_tn_dp_loss.tf \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
# --method Vanilla --mode stream \
# --pretrain_baseline_model_ckpt_path /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss.tf \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 

CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python /home/yangjq/Working/delayed_feedback/src/main.py \
--method FNC --mode stream \
--pretrain_baseline_model_ckpt_path \
 ~/Working/delayed_feedback_release/pretrain_ckpts/pretrain/pretrain \
--data_path /home/yangjq/Datasets/criteo/data.txt \
--data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python \
#  /home/yangjq/Working/delayed_feedback/src/main.py \
# --method FSIW --mode stream \
# --pretrain_baseline_model_ckpt_path \
#  /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss.tf \
# --pretrain_fsiw0_model_ckpt_path \
# /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_FSIW_first_30_train_last_30_test_fsiw0_7days_cd_ce_loss/MLP_FSIW_first_30_train_last_30_test_fsiw0_7days_cd_ce_loss.tf \
# --pretrain_fsiw1_model_ckpt_path \
# /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_FSIW_first_30_train_last_30_test_fsiw1_7days_cd_ce_loss/MLP_FSIW_first_30_train_last_30_test_fsiw1_7days_cd_ce_loss.tf \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python \
#  /home/yangjq/Working/delayed_feedback/src/main.py \
# --method DFM --mode stream \
# --pretrain_dfm_model_ckpt_path \
# ~/Working/delayed_feedback_release/pretrain_ckpts/dfm/dfm \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 

# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python \
#  /home/yangjq/Working/delayed_feedback/src/main.py \
# --method Pretrain --mode stream \
# --pretrain_baseline_model_ckpt_path \
#  /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss.tf \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 


# CUDA_VISIBLE_DEVICES=1 /home/yangjq/anaconda3/envs/tf2/bin/python \
#  /home/yangjq/Working/delayed_feedback/src/main.py \
# --method Oracle --mode stream \
# --pretrain_baseline_model_ckpt_path \
#  /home/yangjq/Working/delayed_feedback/ckpts/benchmark_0.1/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss/MLP_SIG_first_30_train_last_30_test_exp_delay_prtrain_ce_loss.tf \
# --data_path /home/yangjq/Datasets/criteo/data.txt \
# --data_cache_path /home/yangjq/Working/delayed_feedback_release/data_cache 