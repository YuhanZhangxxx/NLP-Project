# 论坛提问草稿

---

## 标题
**用本地 7B LLM 做 battle rap 技巧识别，精度只有人工的 50%，有什么优化思路？**

---

## 背景

我在做一个说唱比赛的 AI 评分系统，比赛规则是 FCPBRL 赛制（类似篮球技术动作的评分体系），每个选手的一轮发言需要被识别出哪些地方用了哪些"技巧"（比如 Slam Dunk = 决定性强力一击，Euro Step = 多层转向双关，Full-Court Shot = 多域跨领域引用链），然后按分值汇总。

文字来源是直播转录，格式大概长这样：

```
I showed him naked.
He basic.
But I showed him Nathan.
He bacon.
The eight engraved in maple wood it look like bacon.
So don't tell me God don't like ugly when he made Viola Davis.
```

这一段对应的技巧是 **Euro Step**（naked → Nathan → bacon 三步转向）+ **Full-Court Shot**（枪 → 神学 → 种族 → 文化引用，四域链接），是非常高分的操作。

---

## 当前方案

### 规则引擎（rule-based）
基于正则 + 关键词的规则引擎，只能可靠检测结构性失误：
- **DD（Double Dribble）**：近似重复行检测（用 difflib.SequenceMatcher，相似度 > 0.80）
- **TVL（Travel/Stumble）**：口吃标记 `[um]`、`[uh]` 等
- 对所有语义类技巧（ES/FCS/SD/SPM 等）完全无效

### 本地 LLM（qwen2.5:7b via Ollama）
把技巧定义和示例写进 system prompt，让模型逐段识别。

**问题：**
1. 识别不了多层文化引用（Nathan→bacon 这条链、Carlton Banks 是 Fresh Prince 里的角色等）
2. 评分过于慷慨，给分没有门槛
3. 把正常的 battle rap 暴力比喻误标为犯规（prompt 里加了说明后有改善但没完全解决）
4. 混淆相近的 skill（把 SPM/Spin Move 判成 FCS，把 DRG/filler 判成 STL）

### Hybrid 方案（现在用的）
规则引擎先跑，结果注入 LLM prompt 作为"已知事实"，LLM 只负责补充语义技巧并确认/否定规则引擎的发现。

**精度对比（两轮实测 vs 人工估算）：**

| 方法 | 轮1（JakkBoy）| 轮2（A. Ward）|
|------|-------------|--------------|
| 规则引擎单独 | -0.75 | +1.50 |
| LLM 单独（7B）| +21.50 | +15.60 |
| Hybrid（7B）| +34.75 | +15.25 |
| **人工估算** | **+24.60** | **+38.60** |

Hybrid 大约达到人工估算的 50-60%，主要差距在正面高分技巧的识别。

---

## 具体问题

**1. 文化知识盲区怎么补？**

Nathan→bacon 这条链需要知道：Nathan 是圣经先知（被揭露之意）→ bacon（熟了/培根）→ 枪的木纹像培根。Viola Davis 是黑人女演员，用来引出种族议题。这类知识 7B 模型不够深，有没有办法在不换更大模型的前提下补充这类领域知识？比如 RAG 挂一个 battle rap 文化知识库？实际效果怎样？

**2. 评分门槛怎么控制？**

7B 模型总是过于慷慨，对普通 bar 也给高分 skill。现在 prompt 里写了"不确定就不给分"但效果有限。有没有更好的 prompt 工程方法，或者用什么结构化输出方式（比如 chain-of-thought 先分析再判断）能提高门槛的准确性？

**3. 多步推理怎么让小模型跟上？**

Euro Step 的识别需要理解 A→B→C 的转向链，Full-Court Shot 需要识别 3+ 个跨域引用被一个主题串起来。这类需要多步推理的任务，7B 模型有没有可能通过 few-shot 或者分步 prompt 做到？还是说这是 7B 的硬上限？

**4. 有没有适合这个任务的开源小模型？**

比较了 qwen2.5:7b（中英双语，但对美国黑人文化理解有限）。有没有在英语文化知识上更强的 7B 模型推荐？或者有没有专门在说唱/流行文化语料上微调过的模型？

---

## 补充信息

- 运行环境：本地 RTX 3070 8GB，部署目标是 RunPod RTX 3080（16GB）
- 推理框架：Ollama
- 语言：英语（美国 battle rap）
- 实时性要求：直播转录后每轮约 15-20 秒内出结果，可接受

如果有人做过类似的特定领域文化理解任务欢迎分享经验，谢谢。
