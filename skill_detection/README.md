# Rap Technique Detection Module

说唱技巧检测模块，包含核心检测逻辑、引擎、技能实现及文档。

## 目录结构

```
skill_detection/
├── loader.py              # 按 skill_id 加载并调用 detect()
├── full_court_chain.py    # 共享 patterns（capture→title→release）
├── rap_techniques.py      # 说唱技巧特征矩阵（供 feats_extra 使用）
├── engines/               # 核心引擎（legacy）
│   ├── punch_strength_engine.py
│   ├── reaction_engine.py
│   └── structure_engine.py
├── 1/, 2/, 3/, 4/, 5/, 6/, 7/, 8/, 9/, 10/, 11/, 17/  # 各技能实现
├── scripts/               # 工具脚本
│   ├── find_misclassified.py
│   └── analyze_misclassified.py
├── test/                  # 测试与示例
├── data/                  # 测试数据
└── docs/                  # 文档
    ├── Rap_Techniques_Usage.md
    ├── Rap_Techniques_Technical.md
    ├── Punch_Strength_Improvements.md
    ├── Test_Interface_Usage.md
    └── Full_Court_Shot_Grammar_Features.md
```

## 快速使用

### 交互式测试
```bash
python skill_detection/test/test_rap_techniques.py --interactive --methods
```

### 数据集测试
```bash
python skill_detection/test/test_rap_techniques.py --test-dataset
```

### 误分类分析
```bash
python skill_detection/scripts/find_misclassified.py
python skill_detection/scripts/analyze_misclassified.py
```

## 主模型集成

```python
import sys
from pathlib import Path
skill_dir = Path(__file__).parent.parent / "skill_detection"
if str(skill_dir) not in sys.path:
    sys.path.insert(0, str(skill_dir))
from rap_techniques import detect_rap_techniques
```

## Completed Skills

| ID | Name | Status |
|----|------|--------|
| 1 | Full Court Shot | Completed |
| 2 | Slam Dunk | Completed |
| 3 | Half Court Shot | Completed |
| 4 | Alley Oop Assist | Completed |
| 5 | And-1 | Completed |
| 6 | Fast Break | Completed |
| 7 | 3 Pointer | Completed |
| 8 | Euro Step | Completed |
| 9 | Steal | Completed |
| 10 | Crossover | Completed |
| 11 | Hook Shot | Completed |
| 17 | Midrange | Completed |

## Rap Techniques (LR features)

1. **Full-Court Shot** (+5.0 pts) – 高风险高回报 bar
2. **Slam Dunk** (+4.25 pts) – 震撼 punch
3. **Half-Court Shot** (+3.75 pts) – 创意风险 hit
4. **Alley-Oop/Assist** (+3.5 pts) – 团队配合
5. **And-1** – 顶着防守得分
6. **Euro Step** – 欧洲步躲避
7. **Steal** – 抢断 flow
8. **Crossover** – 变向 flow
9. **Hook Shot** – 勾手/意外角度

详见 `docs/Rap_Techniques_Usage.md` 和 `docs/Rap_Techniques_Technical.md`。
