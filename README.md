# 四源输入消融 demo

目前只有原图时，先运行 `raw_only_baseline.py`。它用于固定任务定义、检查 API 调用链和建立原图基线，不代表四源输入实验已经完成。

```powershell
python experiments/raw_only_baseline.py `
  --image-dir input_demo/originals `
  --outdir experiments/runs/raw_baseline_001 `
  --env-file llm/.env `
  --limit 20
```

如果已经有人工标签，可以增加 `--gold-dir gold`，其中每个标签文件命名为 `图片文件名.json`。先不要让模型自己生成 gold，否则只能得到自洽性结果，不能得到准确率。

四源实验需要重新生成另外三类输入：

1. `numbered_visualization`：重新训练或调用目标检测器，保存带编号框的图像；
2. `scene_graph`：从检测框、类别、置信度和空间关系确定性生成 JSON；
3. `rule_result`：根据正式采用的安全规则生成初判 JSON，并保留“证据不足”状态。

只有这三类输入和原图一一对应后，才运行下面的四源消融命令。

## 先用数据集自带标注生成第一版四源输入

你的数据集已经有 YOLO 格式标签，因此可以先用验证集标注框生成第一版证据。这个步骤使用真实标注框，目的是单独验证“输入证据如何影响推理”；它不是最终的端到端 YOLO 结果。

在 CMD 中逐行执行：

```bat
cd /d "D:\多模态小论文"
python -m pip install -r llm\requirements.txt
python experiments\prepare_evidence_from_yolo_labels.py ^
  --dataset-root "input_demo\originals\Work at Height Safety Datase(DST2189)" ^
  --split valid ^
  --outdir "experiments\runs\oracle_evidence_001" ^
  --limit 20
```

如果使用 PowerShell，把每行末尾的 `^` 换成反引号 `` ` ``。成功后会生成 `manifest.json`、编号可视化图、场景图和规则结果。

然后运行四源消融：

```bat
python experiments\four_source_ablation_demo.py ^
  --manifest "experiments\runs\oracle_evidence_001\manifest.json" ^
  --outdir "experiments\runs\oracle_ablation_001" ^
  --env-file "llm\.env" ^
  --conditions leave_one_out ^
  --repeats 3 ^
  --temperature 0
```

看到每个样本输出 `valid=True`，并且最后出现 `summary.json` 路径，说明第一轮调用完成。生成的编号图可以在 `experiments\runs\oracle_evidence_001\numbered_visualization` 中查看；如果编号、框或关联明显不对，先停下来修正规则，不要急着解释模型结果。

## 再训练 YOLO 并切换到预测框

完成第一轮 oracle demo 后，再训练目标检测器：

```bat
python -m pip install ultralytics
python experiments\train_yolo.py --data "input_demo\originals\Work at Height Safety Datase(DST2189)\data.yaml" --model yolo11s.pt --epochs 100 --imgsz 640 --batch 8 --device 0 --project "experiments\runs" --name yolo11s_rebuild
```

如果显存不足，把 `--batch 8` 改成 `--batch 4` 或 `--batch 2`。如果没有 NVIDIA GPU，把 `--device 0` 改成 `--device cpu`，但训练会慢很多。

训练结束后，用最佳权重生成预测框证据：

```bat
python experiments\prepare_evidence_from_yolo_predictions.py --weights "experiments\runs\yolo11s_rebuild\weights\best.pt" --image-dir "input_demo\originals\Work at Height Safety Datase(DST2189)\images\valid" --outdir "experiments\runs\pred_evidence_001" --limit 20 --conf 0.25 --device 0
```

再把消融命令中的 manifest 改成：

```bat
python experiments\four_source_ablation_demo.py --manifest "experiments\runs\pred_evidence_001\manifest.json" --outdir "experiments\runs\pred_ablation_001" --env-file "llm\.env" --conditions leave_one_out --repeats 3
```

最终论文需要同时报告两种设置：标注框证据用于隔离多源输入机制，预测框证据用于报告真实端到端系统性能。两者不能混写成同一个结果。

这里的四个输入源定义为：

- `original_image`：原始现场图像；
- `numbered_visualization`：带检测框和人员/目标编号的可视化图；
- `scene_graph`：以人员为中心的场景结构化 JSON；
- `rule_result`：规则层生成的初步结果 JSON。

## 1. 准备 manifest

每张样本用一个 JSON 对象描述。路径相对于 manifest 所在目录解析，也可以使用绝对路径。

```json
[
  {
    "id": "hard_001",
    "original_image": "images/hard_001.jpg",
    "numbered_visualization": "derived/hard_001_numbered.png",
    "scene_graph": "derived/hard_001_scene_graph.json",
    "rule_result": "derived/hard_001_rule_result.json",
    "gold": "gold/hard_001.json"
  }
]
```

`gold` 是人工复核后的参考标签。没有 `gold` 也可以跑 smoke test，但只能报告 JSON 合法率、结构完整率、重复调用一致性和延迟，不能据此声称安全识别准确率提高。

如果需要快速开始人工标注，可以运行 `draft_gold_labels.py` 生成模型辅助草稿和 CSV；如果外部 API 不可用，也可以运行 `fill_local_gold_draft.py`，它只根据本地目标框证据预填明显字段，挂钩、规范使用和最终风险保留为 `unclear`。打开 `gold_review_queue_local.csv` 后，只修改 `human_` 开头的列，并把 `review_status` 改成 `checked`。完成后运行：

```bat
python experiments\finalize_gold_from_review.py --csv "experiments\runs\gold_draft_001\gold_review_queue_local.csv" --outdir "experiments\runs\gold_draft_001\gold"
python experiments\attach_gold_to_manifest.py --manifest "experiments\runs\oracle_evidence_002\manifest.json" --gold-dir "experiments\runs\gold_draft_001\gold" --out "experiments\runs\oracle_evidence_002\manifest_with_gold.json"
```

之后再用 `manifest_with_gold.json` 跑消融，`summary.json` 才会出现基于人工真值的字段准确率和配对差异。

一个最小的 `gold` 文件可以写成：

```json
{
  "overall_risk_level": "high",
  "persons": [
    {
      "person_id": "person_0",
      "work_at_height": "yes",
      "helmet_status": "worn",
      "harness_presence_status": "present",
      "hook_attachment_status": "not_attached",
      "harness_usage_status": "improper_use",
      "final_risk_level": "high"
    }
  ]
}
```

建议第一轮选 20 张左右的困难样本，覆盖：多人、遮挡、小目标、边缘/临边、梯子或脚手架、升降平台、安全带主体可见但挂钩关系不清，以及明确安全和明确违规样本。样本应在正式跑实验前固定，不能只挑模型输出看起来漂亮的图片。

## 2. 运行

```powershell
python experiments/four_source_ablation_demo.py `
  --manifest experiments/demo_manifest.json `
  --outdir experiments/runs/demo_001 `
  --env-file llm/.env `
  --conditions leave_one_out `
  --repeats 3
```

`leave_one_out` 会运行完整四源和四个逐源移除条件。若需要探索所有非空组合，可改用 `--conditions all_subsets`；这会产生 15 个条件，API 成本约为逐源移除设计的三倍。

输出包括：

- `raw_results.jsonl`：每次调用的原始文本、解析 JSON、耗时、结构校验和真值指标；
- `summary.json`：各条件的解析率、结构合法率、真值字段准确率、重复调用一致性、延迟，以及相对于完整输入的配对差异和 bootstrap 区间；
- `config.json`：实验配置和实际使用的输入源组合。

## 3. 如何解读

主结论应来自人工真值上的配对指标：字段准确率、宏平均 F1、人员级风险等级准确率、图像级风险等级准确率，以及错误类型。`summary.json` 里的 `mean_full_minus_ablated` 是一个快速诊断，不替代正式统计检验。

原论文的教师输出一致率可以作为“学生复现教师”的指标保留，但它不是真实安全识别准确率。若教师模型也使用完整四源输入，把教师输出作为唯一基准会形成循环论证；正式论文必须增加独立专家标签，最好由两名具有施工安全知识的标注者独立标注，再对分歧样本进行裁决。
