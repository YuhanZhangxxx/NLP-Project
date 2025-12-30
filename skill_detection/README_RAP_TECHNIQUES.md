# 说唱技巧检测测试工具

## 快速开始

### 1. 安装依赖（如果还没有）
```bash
pip install scipy
# 或者安装所有依赖
pip install -r requirements.txt
```

### 2. 使用测试工具

#### 方式一：交互模式（推荐）
```bash
python src/test_rap_techniques.py
```
然后直接输入要测试的文本，按回车即可看到结果。

#### 方式二：命令行参数
```bash
python src/test_rap_techniques.py "Your spirit left your body in that battle... I pulled the footage — titled it 'Where His Energy Went' and dropped the documentary!"
```

#### 方式三：管道输入
```bash
echo "You only legendary on a T-shirt — every time they remember you, somebody gotta DIE first!" | python src/test_rap_techniques.py
```

## 测试示例

### Full-Court Shot 示例
```bash
python src/test_rap_techniques.py "Your spirit left your body in that battle... I pulled the footage — titled it 'Where His Energy Went' and dropped the documentary!"
```

### Slam Dunk 示例
```bash
python src/test_rap_techniques.py "You only legendary on a T-shirt — every time they remember you, somebody gotta DIE first!"
```

### Half-Court Shot 示例
```bash
python src/test_rap_techniques.py "I flipped your trauma into triumph — the bar so deep the crowd ain't know if they should cheer... or check on you."
```

### Alley-Oop/Assist 示例
```bash
python src/test_rap_techniques.py "Line him up! Cool — I'll rearrange his whole direction like a crooked compass!"
```

## 输出说明

测试工具会显示：
- **🔥 强烈** (≥70%): 技巧非常明显
- **⭐ 明显** (50-70%): 技巧明显存在
- **✓ 存在** (30-50%): 技巧存在但较弱
- **微弱** (<30%): 技巧微弱或不存在

## 集成到模型

说唱技巧检测已经集成到训练模型中。要使用包含技巧特征的模型：

1. **重新训练模型**（包含说唱技巧特征）:
```bash
python src/train_lr_plus.py --csv data/raw/dataset.csv --seed 42
```

2. **使用新模型分析**:
```bash
python src/analyze_song.py --model models/song_lr_v2_plus.joblib --file lyrics.txt
```

分析结果中会包含 `[RAP TECHNIQUES DETECTED]` 部分。

