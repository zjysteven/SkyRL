source /scratch/jingyang/skyrl_venv_docker_test/bin/activate
export PATH="/home/jingyang/.local/bin:$PATH"

set -ex

# ============================================================================
# Quick rollout test with agentdocker-lite (generation only, no training)
#
# Use this to validate the integration works before running full training.
# Compare timing output against run_harbor_gen.sh (Docker baseline).
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="$SCRIPT_DIR/data/harbor"
TRAIN_DATA="['$DATA_DIR/CodeContests']"

CHAT_TEMPLATE_PATH="$(dirname "$0")/../../../skyrl/train/utils/templates/qwen3_acc_thinking.jinja2"
TRIALS_DIR="$SCRIPT_DIR/trials_run_agentdocker"

#----------------
# Infrastructure setup
#----------------
NUM_GPUS=${NUM_GPUS:-4}
ENABLE_RATE_LIMITING=true
TRAJECTORIES_PER_SECOND=5
MAX_CONCURRENCY=16

# agentdocker-lite environment provider
IMPORT_PATH="examples.train_integrations.harbor.agentdocker_lite_environment:AgentDockerLiteEnvironment"

uv run --active --isolated --extra fsdp --extra harbor -m examples.train_integrations.harbor.entrypoints.main_harbor_generate \
  data.train_data=$TRAIN_DATA \
  data.val_data=null \
  harbor_trial_config.trials_dir=$TRIALS_DIR \
  harbor_trial_config.trial_name="dummy" \
  harbor_trial_config.environment.type=null \
  harbor_trial_config.environment.import_path=$IMPORT_PATH \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  generator.inference_engine.served_model_name="Qwen3-8B" \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host="127.0.0.1" \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.sampling_params.max_generate_length=4096 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$NUM_GPUS \
  trainer.policy_mini_batch_size=$NUM_GPUS \
  trainer.logger=console \
  generator.rate_limit.enabled=$ENABLE_RATE_LIMITING \
  generator.rate_limit.trajectories_per_second=$TRAJECTORIES_PER_SECOND \
  generator.rate_limit.max_concurrency=$MAX_CONCURRENCY \
  $@
