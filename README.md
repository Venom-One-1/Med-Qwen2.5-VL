# Med-LLM

基于本地眼科域模型 `Qwen2.5-VL-3B-EyeCoRE-CoTTwoStage` 的判别式眼底多标签 `closed-set` 分类工程。

这个项目的目标是：

- 以本地已经做过眼科域适配的 `Qwen2.5-VL-3B-EyeCoRE-CoTTwoStage` 作为初始化模型
- 输入 `1` 张 `CFP` 和 `1~5` 张 `OCT`
- 输出 `20` 维多标签疾病概率
- 训练框架以 `Llama-Factory` 为底座
- 模型前向、分类头、损失函数和评测逻辑使用当前仓库中的自定义判别式扩展

当前仓库不直接内置完整的 `Llama-Factory` 源码，而是提供一套可以直接接入 `Llama-Factory fork` 的任务实现：

- manifest 构建
- 数据集与 collator
- `Qwen2.5-VL` 判别式分类模型封装
- 训练 / 评测 / 推理 CLI
- 面向 `Llama-Factory fork` 的薄适配层

## 项目结构

```text
Med-LLM/
├─ medllm/
│  ├─ cli/
│  │  ├─ build_manifest.py
│  │  ├─ train.py
│  │  ├─ evaluate.py
│  │  └─ predict.py
│  ├─ llamafactory_ext/
│  │  ├─ integration.py
│  │  └─ README.md
│  ├─ config.py
│  ├─ constants.py
│  ├─ data.py
│  ├─ manifest.py
│  ├─ metrics.py
│  ├─ modeling_qwen25_vl_classifier.py
│  └─ runtime.py
├─ examples/
│  ├─ ophtha_multilabel_train_config.json
│  └─ llamafactory_qwen25_vl_multilabel_lora.yaml
├─ VisualSearch/
├─ Qwen2.5-VL-3B-EyeCoRE-CoTTwoStage/
└─ README.md
```

## 任务定义

- 任务类型：多标签 `closed-set` 分类
- 类别数：`20`
- 样本单位：单眼单次检查
- 图像输入：
  - `1` 张眼底彩照 `CFP`
  - `1~5` 张 `OCT`
- 监督来源：`anno.txt` 第 `3` 列

### 标签映射

标签映射来自 [VisualSearch/mapping_20classes.json](/d:/Projects/Med-LLM/VisualSearch/mapping_20classes.json:1)。

其中：

- `retinal_atrophy`
- `retinal_pigment_epithelial_and_outer_retinal_atrophy`
- `choroidal_atrophy`

统一映射到同一个类别 id `13`。

## 数据组织

外部图像根目录默认是：

```bash
/home/sqw/VisualSearch/mm_linyan/imgdata
```

`anno.txt` 每一行至少包含 3 列：

1. `source_key`
2. `OCT` 索引列表
3. 标签名列表

### 图像路径规则

- `source_key` 直接等于样本目录相对路径
- 样本目录通过 `image_root/source_key` 定位
- `CFP` 文件名固定为 `<样本ID>.fundus.jpg`
- `OCT` 文件按第二列索引解析，优先使用 `<样本ID>_{k:03d}.jpg`
- `image_paths` 顺序固定为：
  - `fundus` 在前
  - 后续 `OCT` 严格按 `anno.txt` 第二列给出的顺序追加

### manifest 格式

manifest 为 `jsonl`，每条记录包含：

- `sample_id`
- `split`
- `source_key`
- `image_paths`
- `label_names`
- `label_ids`
- `label_vec`

示例：

```json
{
  "sample_id": "sample-001-OD",
  "split": "train",
  "source_key": "a/b/sample-001-OD",
  "image_paths": [
    "/home/sqw/VisualSearch/mm_linyan/imgdata/a/b/sample-001-OD/sample-001-OD.fundus.jpg",
    "/home/sqw/VisualSearch/mm_linyan/imgdata/a/b/sample-001-OD/sample-001-OD_001.jpg",
    "/home/sqw/VisualSearch/mm_linyan/imgdata/a/b/sample-001-OD/sample-001-OD_010.jpg"
  ],
  "label_names": ["macular_edema", "retinal_vein_occlusion"],
  "label_ids": [1, 9],
  "label_vec": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
```

## 模型设计

当前实现采用你确认过的判别式路线：

`LM Decoder 最后一层 image-region embeddings -> 分层 mean-pooling -> MLP 分类头`

具体做法：

- 基座模型：本地 `Qwen2.5-VL-3B-EyeCoRE-CoTTwoStage`
- `Vision Encoder`：冻结
- `projector / merger`：保持可训练
- `LM Decoder`：注入普通 `LoRA`
- 分类头：`2048 -> 2048 -> 20`
- 激活函数：`GELU`
- dropout：`0.1`
- 损失函数：`BCEWithLogitsLoss`

分层 pooling 的逻辑是：

1. 对每张图像的 image-region tokens 做一次 mean-pooling
2. 再对 `CFP + OCT` 的图像级 embedding 做第二层聚合
3. 得到样本级融合表示后接 `MLP` 输出 `20` 维 logits

## 安装

建议使用 Python `3.10+`。

先安装依赖：

```bash
pip install -r requirements.txt
```

如果你要在 GPU 环境里训练，请根据你的 CUDA 版本安装匹配的 `torch` 构建。

当前仓库没有直接 vendor 完整的 `Llama-Factory`，因此推荐两种使用方式：

1. 先在当前仓库安装依赖并运行 manifest / 本地训练脚本
2. 将 `medllm/` 目录接入你自己的 `Llama-Factory fork`

## 快速开始

### 1. 生成 manifest

```bash
python -m medllm.cli.build_manifest \
  --image-root /home/sqw/VisualSearch/mm_linyan/imgdata \
  --mapping-path VisualSearch/mapping_20classes.json \
  --output-dir outputs/manifests \
  --max-oct-images 5
```

如果你当前只想检查标注逻辑，不校验真实图像文件，可以加：

```bash
--skip-image-validation
```

生成结果默认包括：

- `outputs/manifests/train.jsonl`
- `outputs/manifests/val.jsonl`
- `outputs/manifests/test.jsonl`
- `outputs/manifests/summary.json`

### 2. 训练

使用示例配置：

```bash
python -m medllm.cli.train --config examples/ophtha_multilabel_train_config.json
```

默认关键配置位于：

- [examples/ophtha_multilabel_train_config.json](/d:/Projects/Med-LLM/examples/ophtha_multilabel_train_config.json:1)

默认训练设置：

- `LoRA`
- `bf16`
- 双卡各 `batch_size=1`
- `gradient_accumulation_steps=8`
- `epoch=12`
- `Macro-F1` 选最佳 checkpoint

### 3. 评测

```bash
python -m medllm.cli.evaluate \
  --config examples/ophtha_multilabel_train_config.json \
  --checkpoint outputs/qwen25_vl_multilabel/best_model.pt \
  --manifest outputs/manifests/test.jsonl \
  --output outputs/qwen25_vl_multilabel/test_metrics.json
```

评测输出包括：

- `Macro-F1`
- `Micro-F1`
- `mAP`
- `Macro-AUROC`
- 每类 `F1 / Precision / Recall / Specificity / AP / AUROC / threshold`

### 4. 推理

```bash
python -m medllm.cli.predict \
  --config examples/ophtha_multilabel_train_config.json \
  --checkpoint outputs/qwen25_vl_multilabel/best_model.pt \
  --image /path/to/sample.fundus.jpg \
  --image /path/to/sample_001.jpg \
  --image /path/to/sample_010.jpg
```

输出字段包括：

- `probabilities`
- `pred_vec`
- `pred_labels`
- `thresholds`

## 与 Llama-Factory 的关系

这个仓库的定位不是替代 `Llama-Factory`，而是给 `Llama-Factory` 提供任务特定的判别式扩展。

推荐的接入方式是：

1. 在你的 `Llama-Factory fork` 中注册新的 stage 或任务类型
2. 复用当前仓库中的：
   - `medllm.manifest`
   - `medllm.data`
   - `medllm.modeling_qwen25_vl_classifier`
   - `medllm.metrics`
   - `medllm.llamafactory_ext.integration`
3. 将原始生成式 loss 替换为分类 logits 上的 `BCEWithLogitsLoss`
4. 验证时执行逐类阈值搜索，并按 `macro_f1` 保存最佳 checkpoint

一个 launcher 风格的参考配置在：

- [examples/llamafactory_qwen25_vl_multilabel_lora.yaml](/d:/Projects/Med-LLM/examples/llamafactory_qwen25_vl_multilabel_lora.yaml:1)

## 主要模块说明

- [medllm/manifest.py](/d:/Projects/Med-LLM/medllm/manifest.py:1)
  - 负责 `anno.txt -> manifest`
  - 处理 `fundus` 和 `OCT` 路径解析
  - 负责过滤缺图、坏样本、未知标签

- [medllm/data.py](/d:/Projects/Med-LLM/medllm/data.py:1)
  - 提供 manifest dataset
  - 提供多图输入 collator
  - 生成固定模态提示词

- [medllm/modeling_qwen25_vl_classifier.py](/d:/Projects/Med-LLM/medllm/modeling_qwen25_vl_classifier.py:1)
  - 封装 `Qwen2.5-VL` 判别式模型
  - 实现 image-region token 提取与分层 pooling
  - 实现 `MLP` 分类头

- [medllm/runtime.py](/d:/Projects/Med-LLM/medllm/runtime.py:1)
  - 提供训练、评测、推理逻辑
  - 支持逐类阈值搜索
  - 兼容 `torchrun` 风格 DDP

- [medllm/llamafactory_ext/integration.py](/d:/Projects/Med-LLM/medllm/llamafactory_ext/integration.py:1)
  - 为 `Llama-Factory fork` 提供模型、dataloader 和评测对接函数

## 注意事项

- 当前环境如果没有安装 `torch / transformers / peft / pillow`，训练和推理无法真正运行
- 当前仓库可以独立完成 manifest 构建和静态代码检查
- 真正的双卡训练建议在 Linux + CUDA 环境中执行
- `predict` 和 `train` 默认都依赖本地模型目录 `Qwen2.5-VL-3B-EyeCoRE-CoTTwoStage`
- 这个项目第一版只做样本级多标签分类，不做双眼级、病例级和时序建模

## 后续建议

如果你接下来继续迭代，比较值得优先尝试的方向是：

- 在线接入完整 `Llama-Factory fork`
- 补齐真实训练环境后的端到端 smoke test
- 在线性头和两层 `MLP` 头之间做 ablation
- 对 pooling 策略做更细的模态权重实验
- 加入更完整的错误样本分析和可视化输出
