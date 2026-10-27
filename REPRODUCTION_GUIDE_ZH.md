# SkillRL 复现手册

本文把论文《SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning》与本仓库代码对应起来。论文中的指令、提示词和实验设置是复现对象；真正执行时只运行下面列出的仓库脚本。

## 先确定复现目标

论文主实验由三条实验线组成：

| 实验线 | 环境/数据 | 论文指标 | 仓库入口 |
| --- | --- | --- | --- |
| ALFWorld | 文本交互，6 类任务 | 每类 success rate 和 overall success | `examples/grpo_trainer/run_alfworld_skills.sh` |
| WebShop | 文本网页购物 | average score、success rate | `examples/grpo_trainer/run_webshop_skills.sh` |
| Search | NQ、HotpotQA 训练，7 个 QA 集合评测 | 各数据集 EM/accuracy 和平均值 | `examples/grpo_trainer/run_search_skills.sh`、`examples/search/` |

论文的完整顺序是：

```text
基础模型 rollout
  -> 成功/失败轨迹
  -> 轨迹解析与去重
  -> teacher(o3) 生成 memory/skill bank
  -> teacher 生成 skill-aware SFT 数据
  -> Qwen2.5-7B-Instruct 冷启动 SFT
  -> 以 SFT checkpoint 为 policy 和 reference 的 GRPO
  -> 每 5 个验证步骤分析失败轨迹并扩展 SkillBank
  -> 保存 checkpoint、指标、更新后的 skill bank
```

仓库已经提交了 ALFWorld、WebShop、Search 的初始 skill bank，因此第一次复现建议先跳过昂贵的 rollout、teacher distillation 和 SFT 数据生成，直接使用发布的 SFT checkpoint 加初始 skill bank 跑 RL。完整复现实验再从头生成数据。

## 论文参数与代码对应

论文使用 Qwen2.5-7B-Instruct，o3 作为 teacher。冷启动 SFT 使用学习率 `1e-4`、batch size `16`、3 epochs，样本数约为 ALFWorld 7,500、WebShop 2,400。RL 使用 GRPO，学习率 `1e-6`、group size `8`、KL loss coefficient `0.01`、invalid-action penalty coefficient `0.1`，最大 prompt/response 长度分别为 6,000/1,024，训练 150 个 epoch。论文正文的 implementation details 写的是 RL batch size 16，Appendix D.1 表格写的是 64；仓库 ALFWorld/WebShop 脚本默认 train data size 为 16，服务器上应记录实际 effective batch 并保持 `train_batch_size * env.rollout.n` 能被 GPU 数整除。

仓库的 ALFWorld/WebShop skill 脚本已经带有动态 skill 更新、top-k=6、阈值 0.4、每轮最多新增 3 个 skill 和每 5 步验证。脚本为了适配不同服务器，已改为按脚本位置计算仓库根目录，数据目录可通过环境变量覆盖。Search 脚本原本是“只验证、不训练”的配置，现已按论文参数改为 150 epoch 训练入口；训练数据合并和七个评测集的拆分仍需要在服务器上按论文数据准备。

## 服务器上你需要准备的东西

1. Linux 服务器，建议 8×H100 80GB；论文总耗时约 30 小时/实验。单卡或消费级显卡只能做环境、skill retrieval 和数据管线检查，不能复现论文规模的 7B GRPO。
2. Python 3.10 或 3.11、CUDA 与 PyTorch 对应的 GPU 版本。不要用当前 Windows 的 Python 3.14 环境。
3. Qwen2.5-7B-Instruct 基础模型，或论文发布的三个任务 SFT checkpoint：
   - `Jianwen/Alfworld-7B-SFT`
   - `Jianwen/Webshop-7B-SFT`
   - `Jianwen/Search-7B-SFT`
4. 训练依赖：仓库 `requirements.txt`、与服务器 CUDA 匹配的 `vllm`、`flash-attn`，以及仓库本身的 editable install。推荐先按 README 安装，再根据 vLLM/PyTorch 兼容性锁版本。
5. ALFWorld 数据：安装 `alfworld`、`gymnasium==0.29.1`、`stable-baselines3==2.6.0`，执行 `alfworld-download -f`，并设置 `ALFWORLD_DATA`。
6. WebShop 数据：在 `agent_system/environments/env_package/webshop/webshop` 执行 `bash setup.sh -d all`。它需要 Google Drive 数据、Java 11、spaCy 模型和索引构建。
7. Search 数据：下载 Search-R1 处理后的 train/test parquet；另外下载 wiki-18 corpus/index，并启动 `examples/search/retriever/retrieval_server.py`。检索服务地址必须与训练脚本中的 `env.search.search_url` 一致。
8. teacher API：可以使用你自己的 OpenAI 兼容接口，不需要论文作者的 Azure 账号。设置 `LLM_API_KEY`、`LLM_BASE_URL`（通常以 `/v1` 结尾）和 `LLM_MODEL`。SFT 数据生成脚本还要设置兼容的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`；如果使用 Azure，继续设置 `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT` 和可选的 `AZURE_OPENAI_API_VERSION`。
9. 可选的 Weights & Biases：设置 `WANDB_API_KEY`，否则把脚本里的 logger 改成 `['console']`。

## 推荐的服务器执行顺序

### 1. 安装代码和依赖

```bash
git clone https://github.com/aiming-lab/SkillRL.git
cd SkillRL
conda create -n skillrl python=3.10 -y
conda activate skillrl
pip install -r requirements.txt
pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
pip install 'openai>=1.0.0'
pip install faiss-cpu sentence-transformers
```

配置自有 OpenAI 兼容接口时，先在当前 shell 导出变量（不要把真实 Key 写进脚本或提交到 Git）：

```bash
export LLM_API_KEY="你的 API Key"
export LLM_BASE_URL="https://api.aicodemirror.ai"
export LLM_MODEL="gpt-5.6-terra"
export OPENAI_API_KEY="$LLM_API_KEY"
export OPENAI_BASE_URL="$LLM_BASE_URL"
export OPENAI_MODEL="$LLM_MODEL"
```

确认服务商支持 OpenAI Chat Completions 格式后，可以先发一个短请求验证模型名、地址和 Key；skill 生成与动态更新会使用 `LLM_*`，SFT 数据生成会使用 `OPENAI_*`。

安装后先检查：

```bash
python - <<'PY'
import torch, transformers, ray
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('transformers', transformers.__version__, 'ray', ray.__version__)
PY
nvidia-smi
```

### 2. 先验证已有 skill bank

三份初始 skill bank 在：

```text
memory_data/alfworld/claude_style_skills.json
memory_data/webshop/claude_style_skills.json
memory_data/search/claude_style_skills_search.json
```

模板检索不需要 embedding 模型。验证时要确认：能加载 JSON、能识别任务类型、能输出 general/task-specific/query-specific skills、能格式化成 prompt。Search bank 使用 `query_type_skills` schema，运行时已在 `SkillsOnlyMemory` 中兼容为 `task_specific_skills`。

### 3. 选择 SFT 来源

推荐路线是下载论文发布的任务 SFT checkpoint，直接把它作为 `MODEL_PATH`。从头做冷启动 SFT 时：

1. 先用基础模型在目标环境生成轨迹，保存成 `examples/sft_data_generation/README.md` 规定的 txt 格式。
2. 运行对应的 `examples/sft_data_generation/run_alfworld.sh`、`run_webshop.sh` 或 `run_search.sh`。
3. 这些脚本依次执行 parser、重复轨迹去除、per-trajectory memory、skill bank 聚合、o3 distillation 和 ShareGPT/alpaca 展平。
4. WebShop 还要运行 action validator；ALFWorld distillation 会补 synthetic done step。
5. 将输出 JSON 转为 LLaMA-Factory 的 alpaca 数据格式，按论文超参训练 3 epochs，得到 `MODEL_PATH`。

如果没有自己的 rollout，直接使用仓库内已经提交的 `memory_data/*/generated_memories*.json` 和 `claude_style_skills*.json`，或者使用 Hugging Face 发布的 SFT 数据/任务 SFT checkpoint。这样可以先验证 RL 代码，再补做 teacher 数据生成。

### 4. ALFWorld RL

```bash
export MODEL_PATH=/path/to/Alfworld-7B-SFT
export ALFWORLD_DATA=/path/to/alfworld_data
export WANDB_API_KEY=...
bash examples/grpo_trainer/run_alfworld_skills.sh vllm
```

默认脚本是小批量的仓库实现（train 16、validation 64、group 8），适合先做启动检查；论文规模需要根据 GPU 显存把 train batch、micro batch 和并行度调到服务器可承受的值，同时保留 lr、KL、top-k、阈值和 150 epoch。动态更新版本是 `run_alfworld_skills_train_update.sh`，它从训练失败轨迹更新 skill；论文主描述对应验证失败更新，因此优先使用 `run_alfworld_skills.sh`。

### 5. WebShop RL

```bash
export MODEL_PATH=/path/to/Webshop-7B-SFT
bash examples/grpo_trainer/run_webshop_skills.sh vllm
```

先用 `webshop.use_small=True` 做启动检查；论文结果使用完整 WebShop 数据时改成 `env.webshop.use_small=False`，并确认 `items_shuffle.json`、`items_ins_v2.json` 和索引存在。WebShop 的环境奖励由 wrapper 转为成功 10、失败 0，同时保留原始 task score 供统计。

### 6. Search RL/评测

先准备 parquet：

```bash
python examples/data_preprocess/preprocess_search_r1_dataset.py \
  --hf_repo_id PeterJinGo/nq_hotpotqa_train \
  --local_dir $HOME/data/searchR1_processed_direct
```

再下载 wiki-18 index/corpus 并启动检索服务。`run_search_skills.sh` 已按论文参数改成 150 epoch 的训练入口，会读取 `SEARCH_DATA_DIR` 和 `env.search.search_url`。论文复现需要额外准备 NQ+HotpotQA 的训练合并 parquet、七个数据集的验证 parquet，并在评测时分别替换 `data.val_files`，计算 Table 2 的平均值。若只想做评测，可以在脚本末尾追加 `trainer.total_epochs=0 trainer.val_before_train=True` 覆盖训练配置。不要把只在 NQ/HotpotQA 上训练后的单次验证结果直接当作 Table 2 的七数据集平均值。

## 动态 Skill Evolution 的实际实现

验证结束后，`RayPPOTrainer` 按 task-specific success rate 判断是否低于 `update_threshold`。低于阈值时收集失败轨迹，调用 `agent_system/memory/skill_updater.py` 的 o3 updater，生成 `dyn_XXX` skill，加入训练环境的 memory，并保存到 `trainer.default_local_dir/updated_skills_step*.json`。训练环境更新版本由 `update_skills_from_train=True` 控制。验证更新默认只写训练环境，避免把针对验证集调出的 skill 直接注入后续验证。

## 每一步应该保存的证据

每条实验线至少保留：安装版本和 `nvidia-smi`、完整 Hydra 配置、初始 skill bank、SFT checkpoint 路径、训练日志、每 5 步验证指标、最终 checkpoint、`updated_skills_step*.json`、以及按任务类别汇总的成功率。论文表格的结果来自这些验证/测试日志，不要只保存最终 overall score。

## 常见卡点

- `ray`、`vllm` 或 `flash-attn` 导入失败：通常是 Python/CUDA/PyTorch 版本组合不匹配，先锁 Python 3.10/3.11 和服务器 CUDA 对应的 PyTorch，再装 vLLM/flash-attn。
- ALFWorld 找不到数据：检查 `ALFWORLD_DATA` 和 `agent_system/environments/env_package/alfworld/configs/config_tw.yaml` 中的路径变量。
- WebShop 启动后找不到文件或索引：重新执行 `setup.sh -d all`，确认 Java 11、spaCy 模型和 `data/` 文件齐全。
- Search 返回连接错误：先单独 `curl` 检查检索服务，再运行 trainer；训练脚本中的地址必须与服务端口一致。
- 动态更新没有新增 skill：检查 `enable_dynamic_update=True`、success rate 是否低于 0.4，以及 Azure API 三个环境变量；更新失败不会中断训练，只会打印 `[SkillUpdate]` 日志。
- 运行时显存不足：先降低 `data.train_batch_size`、`ppo_micro_batch_size_per_gpu`、`max_num_seqs` 和 vLLM `gpu_memory_utilization`，保持 `data.train_batch_size * env.rollout.n` 能被总 GPU 数整除。

## 本次我已经补齐的仓库问题

1. 新增 `skill_generation/llm_client.py`，skill 生成和 RL 动态更新现在支持 `LLM_API_KEY + LLM_BASE_URL + LLM_MODEL` 的任意 OpenAI 兼容接口；原 Azure 变量仍然兼容。缺少配置时会给出明确错误。
2. README 中 Search skill bank 生成命令原来误指向 WebShop memory，已改为 `memory_data/search/generated_memories_search.json` 和 `memory_data/search/claude_style_skills_search.json`。
3. ALFWorld/WebShop/Search 的 GRPO 脚本现在从脚本位置解析仓库根目录，技能 JSON 不依赖用户目录名；ALFWorld/WebShop 的 parquet 路径可用 `DATA_DIR` 覆盖，Search 可用 `SEARCH_DATA_DIR` 覆盖，脚本会在缺少 `MODEL_PATH` 时立即报错。
4. `SkillsOnlyMemory` 现在兼容 Search 的 `query_type_skills` schema，否则 Search 任务只能读 general skills，无法读取四类 query-specific skills。

这些改动只涉及配置读取、路径可移植性和 schema 兼容，没有改动论文的 GRPO 目标函数或环境奖励定义。当前没有替你生成 SFT checkpoint、下载外部环境数据或调用 Azure/o3，因为这些步骤需要服务器存储、网络和你的 API 凭据。
