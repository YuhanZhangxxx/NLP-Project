# 说唱技巧识别模块

本文件夹包含所有说唱技巧识别相关的代码、数据和文档，与AI主模型分离。

## 文件结构

```
skill_detection/
├── rap_techniques.py              # 核心技巧识别模块
├── test_rap_techniques.py         # 测试工具（交互/批量/数据集测试）
├── rap_techniques_README.md       # 技术文档
├── README_RAP_TECHNIQUES.md       # 使用说明
├── data/                          # 测试数据
│   ├── full_court_shot_correct.txt
│   ├── full_court_shot_incorrect.txt
│   └── full_court_shot_test.csv
└── docs/                          # 技术文档
    ├── Full_Court_Shot_Detection_Explanation.md
    ├── Full_Court_Shot_Grammar_Features.md
    └── Test_Interface_Usage.md
```

## 使用方法

### 交互测试
```bash
python skill_detection/test_rap_techniques.py --interactive --methods
```

### 数据集测试
```bash
python skill_detection/test_rap_techniques.py --test-dataset
```

### 命令行测试
```bash
python skill_detection/test_rap_techniques.py "your text here" --methods
```

## 在主模型中使用

主模型（`src/` 目录下的文件）通过以下方式导入：

```python
import sys
from pathlib import Path
skill_dir = Path(__file__).parent.parent / "skill_detection"
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))
from rap_techniques import detect_rap_techniques
```

## 识别的技巧

1. **Full-Court Shot** (+5.0分) - 高风险高回报的bar
2. **Slam Dunk** (+4.25分) - 震撼性的重击
3. **Half-Court Shot** (+3.75分) - 创意风险命中
4. **Alley-Oop/Assist** (+3.5分) - 多人配合

## 击溃语义识别方法

- **方法1**: 关键词库识别（基于预定义关键词和模式）
- **方法2**: 词性识别（基于NLTK词性标注）

两种方法使用OR逻辑并列，只要任一方法识别到击溃语义即可。

