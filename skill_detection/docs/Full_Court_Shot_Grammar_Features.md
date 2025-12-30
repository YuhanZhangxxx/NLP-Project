# Full-Court Shot 语法特征分析

## 概述

Full-Court Shot 在英语里是一种"3拍连招句式"，语法上非常规律，像在报导一件事的流程。本文档记录了基于语言学分析的语法特征检测方法。

## 核心语法特征

### 1. 三连动词句式（最强信号）

**核心句式骨架：**
```
I + VERB1 + OBJECT1 + VERB2 + OBJECT2 + VERB3 + OBJECT3
```

**典型动词组合：**
- **VERB1 (捕获/创建类)**: `pulled`, `clipped`, `grabbed`, `captured`, `caught`, `snapped`, `recorded`, `ran`, `printed`, `wrote`, `filed`, `issued`, `opened`, `logged`, `tracked`, `made`, `created`, `pushed`, `posted`, `built`, `constructed`, `assembled`, `erected`
- **VERB2 (命名类)**: `titled`, `named`, `called`, `labeled`, `tagged` + `it/this/that/the`
- **VERB3 (发布类)**: `dropped`, `released`, `unveiled`, `debuted`, `aired`, `broadcast`, `streamed`, `premiered`, `uploaded`, `published`, `opened`, `launched`, `presented`

**语法特点：**
- 同一主语 `I`
- 连续三个过去式及物动词
- 中间夹一个命名动作（`titled/named/called`）
- 形成"我做了三步"的节奏感

**示例：**
```
I pulled the replay titled it Where His Heart Went and aired the documentary
I clipped the moment named it Live Disappearance and released the case file
I built the museum display titled it The Fall Of The Front and opened the exhibit
```

### 2. 两段结构

**结构模板：**
```
[时间/条件开头] + [对手崩溃状态] + I + [三连动作链]
```

**第一段（对手状态）：**
- 时间框架：`By the end of the round`, `The moment`, `After that round`, `Mid battle`
- 对手崩溃：`your X was Y`, `your X vanished/disappeared`, `your X got stolen/deleted`

**第二段（制作流程）：**
- `I + VERB1 + OBJ1 + VERB2 + OBJ2 + VERB3 + OBJ3`

**示例：**
```
By the end of the round your hype was just an empty chair I built the museum display titled it The Fall Of The Front and opened the exhibit
After that round the whole room watched your confidence vanish I pulled the replay titled it Where His Heart Went and aired the documentary
```

### 3. 连接词偏好

**常见连接方式：**
- `and and and` - 连续and，强化流程感
- `and then` - 时间顺序
- `so I` - 因果关系
- `and I` - 简单连接

**语法作用：**
- 强化"一步接一步"的流程感
- 让句子像流水线作业
- 最朴素的连接方式，不花哨

### 4. 时间框架开头

**常见时间框架：**
- `By the end of` - 结束时
- `The moment` - 那一刻
- `As soon as` - 一...就
- `After that round` - 那轮之后
- `Mid battle` - 战斗中
- `Soon as` - 一...就
- `The second` - 那一秒
- `When the` - 当...时
- `That round` - 那轮
- `One beat switch later` - 一个节拍切换后

**语法作用：**
- 让句子更像新闻播报
- 强调"事件发生后我立刻处理"
- 状语从句或时间短语开头

### 5. 标题语法特征

**常见标题格式：**
- `Where His X Went` - 去向类
- `The Day He Y` - 时间类
- `Live Noun` - 现场类
- `Missing Since Noun` - 失踪类
- `X Autopsy` - 解剖类
- `X Heist` - 抢劫类
- `The Fall Of X` - 衰落类
- `Exhibit A` - 证据类
- `The Exact Frame` - 精确帧类

**语法特点：**
- 名词短语
- 从句缩写式标题
- 像媒体标题一样可发布
- 通常大写开头（但在无标点文本中可能不明显）

### 6. 人称配置

**典型配置：**
- `you/your` - 作为被击溃对象（受害者）
- `I` - 作为导演/记者/法官（制作人）

**语法作用：**
- 天然制造"我记录并公开你的失败"的压迫感
- 视角对比：你失败 vs 我制作内容

## 检测算法实现

### 语法特征检测（新增）

在 `detect_full_court_shot()` 函数中添加了以下语法特征检测：

1. **三连动词句式检测** (+0.3分)
   - 检测 `I + VERB1 + ... + VERB2 (titled/named/called) + ... + VERB3` 模式
   - 允许中间有连接词和少量其他词

2. **时间框架开头检测** (+0.1分)
   - 检测句首的时间框架短语

3. **两段结构检测** (+0.15分)
   - 检测"时间框架 + 对手状态 + I + 动作"的结构

4. **连接词偏好检测** (+0.05-0.1分)
   - 检测 `and and and`, `and then`, `so I`, `and I` 等

5. **人称配置检测** (+0.1分)
   - 检测同时包含 `you/your` 和 `I` 的配置

6. **标题语法特征检测** (+0.08-0.15分)
   - 检测标题的语法模式（名词短语、从句缩写式等）

## 效果

添加语法特征检测后：
- **正确例子平均分数**: 68.5% → 90.8% (提升22.3%)
- **多个正确例子达到100%**
- **准确率保持97.5%**
- **错误例子平均分数**: 0.8% (仍然很低，无误报)

## 最强语法信号

**用户建议的最强语法信号：**
> 同一主语 I 连续三个过去式及物动词，且中间夹一个 titled/named/called 的命名动作

这正是我们检测的核心模式，也是Full-Court Shot最典型的语法特征。


