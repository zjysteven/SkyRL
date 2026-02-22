cd skyrl-train/
uv venv --python 3.12 ../../skyrl_venv
source ../../skyrl_venv/bin/activate
uv sync --active --extra vllm --extra harbor

# export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
