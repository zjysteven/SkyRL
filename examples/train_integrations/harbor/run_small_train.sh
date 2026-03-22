#!/bin/bash
source /scratch/jingyang/skyrl_venv_docker_test/bin/activate
export PATH="/home/jingyang/.local/bin:$PATH"

set -ex

# ============================================================================
# Small-scale RL training for A/B comparison: agentdocker-lite vs Docker
#
# Usage:
#   MODE=adl    CUDA_VISIBLE_DEVICES=1 bash run_small_train.sh
#   MODE=docker CUDA_VISIBLE_DEVICES=5 bash run_small_train.sh
# ============================================================================

MODE="${MODE:-adl}"  # "adl" or "docker"
NUM_GPUS=${NUM_GPUS:-2}
SEED=42

SCRIPT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"
CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"

RUN_NAME="small-train-${MODE}"
TRIALS_DIR="$SCRIPT_DIR/$RUN_NAME/trials"
CKPTS_DIR="$SCRIPT_DIR/$RUN_NAME/ckpts"
LOG_DIR="/tmp/skyrl-logs/$RUN_NAME"

# Environment config
if [ "$MODE" = "adl" ]; then
    ENV_ARGS="harbor_trial_config.environment.type=null harbor_trial_config.environment.import_path=examples.train_integrations.harbor.agentdocker_lite_environment:AgentDockerLiteEnvironment"
    HTTP_PORT=8000
else
    ENV_ARGS="harbor_trial_config.environment.type=docker"
    HTTP_PORT=8001
fi

uv run --active --isolated --extra fsdp --extra harbor -m examples.train_integrations.harbor.entrypoints.main_harbor \
  data.train_data=$TRAIN_DATA \
  data.val_data=null \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  harbor_trial_config.trial_name="small" \
  $ENV_ARGS \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  trainer.ckpt_path=$CKPTS_DIR \
  trainer.log_path=$LOG_DIR \
  trainer.seed=$SEED \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.loss_reduction=seq_mean_token_sum_norm \
  trainer.algorithm.grpo_norm_by_std=false \
  trainer.algorithm.use_kl_loss=false \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.inference_engine.engine_init_kwargs.max_model_len=8192 \
  generator.inference_engine.engine_init_kwargs.enable_log_requests=false \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.algorithm.max_seq_len=8192 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.n_samples_per_prompt=2 \
  generator.sampling_params.max_generate_length=4096 \
  generator.apply_overlong_filtering=true \
  generator.inference_engine.gpu_memory_utilization=0.4 \
  trainer.logger=console \
  trainer.project_name=harbor \
  trainer.run_name=$RUN_NAME \
  trainer.resume_mode=none \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=$HTTP_PORT \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=5 \
  generator.rate_limit.max_concurrency=16 \
  "$@"
