# AgentDocker-Lite Integration for SkyRL Harbor

Drop-in replacement for Harbor's `DockerEnvironment` using Linux namespace sandboxes (agentdocker-lite) for ~20x faster container lifecycle.

## Created Files

### 1. Core Implementation: `agentdocker_lite_environment.py`

Implements Harbor's `BaseEnvironment` interface, internally using agentdocker-lite's `Sandbox`:

- **`start()`** — Builds image from Dockerfile, exports to rootfs (with content-hash caching), creates Sandbox (~10ms)
- **`exec()`** — Executes commands via persistent shell with `cwd` and `env` support (~11ms/command)
- **`upload_file/dir()`** — Direct host rootfs filesystem operations (zero-copy)
- **`download_file/dir()`** — Same as above
- **`stop()`** — Deletes sandbox (~2ms)
- **Rootfs caching** — Cached by Dockerfile content hash; all CodeContests tasks share a single rootfs
- **Timing instrumentation** — Automatically collects per-operation latencies, prints summary at process exit

### 2. Training Script: `run_codecontest_agentdocker.sh`

Identical hyperparameters to `run_codecontest.sh`, with only the environment provider changed:

```bash
harbor_trial_config.environment.type=null
harbor_trial_config.environment.import_path=examples.train_integrations.harbor.agentdocker_lite_environment:AgentDockerLiteEnvironment
```

### 3. Quick Validation Script: `run_harbor_gen_agentdocker.sh`

Generation-only mode, runs 10 samples to quickly verify the integration works.

### 4. Dependency Configuration: `pyproject.toml`

Added `agentdocker-lite` as a dependency under the `harbor` optional dependency group, pointing to the local repo at `/scratch/jingyang/agentdocker-lite`.

## Test Steps

### Step 1: Quick Validation (verify integration works)

```bash
cd /scratch/jingyang/SkyRL_docker_test
bash examples/train_integrations/harbor/run_harbor_gen_agentdocker.sh
```

Check the output for `AgentDockerLite sandbox started` log lines and the final Timing Summary printed at process exit.

### Step 2: A/B Training Comparison

```bash
# Baseline: Docker
bash examples/train_integrations/harbor/run_codecontest.sh

# Experiment: agentdocker-lite
bash examples/train_integrations/harbor/run_codecontest_agentdocker.sh
```

### Step 3: Compare Results

**Accuracy / Loss (Goal 1):**
- Compare `codecontest` vs `codecontest-agentdocker` runs in wandb
- Loss curves should follow the same trend
- Final eval reward/accuracy should be within statistical variance

**Speed (Goal 2):**
- Compare the Timing Summary printed at process exit:
  - `start` — sandbox creation time (Docker ~271ms vs agentdocker-lite ~10ms)
  - `exec` — per-command latency (Docker ~22ms vs agentdocker-lite ~11ms)
  - `stop` — cleanup time (Docker ~104ms vs agentdocker-lite ~2ms)
- Compare end-to-end rollout time per training step in wandb metrics
