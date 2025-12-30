# 说唱技巧检测测试工具使用指南

## 概述

`skill_detection/test_rap_techniques.py` 是一个统一的测试工具，提供两种独立的接口：
1. **交互模式** - 用于手动测试和调试
2. **数据集测试** - 用于批量评估准确率

## 接口说明

### 1. 交互模式（默认）

**启动方式：**
```bash
# 默认启动交互模式
python skill_detection/test_rap_techniques.py

# 显式指定交互模式
python skill_detection/test_rap_techniques.py --interactive
python skill_detection/test_rap_techniques.py -i
```

**功能：**
- **单条测试**：直接输入文本，按回车测试
- **批量测试**：输入多行，每行以数字开头（如: `1. 文本`），自动进入批量模式
- **退出**：输入 `quit` 或 `exit`

**使用示例：**
```
请输入文本（或批量测试的第一行）: After that round I pulled the clip titled it Where Your Voice Went and dropped the documentary
[显示结果]

请输入文本（或批量测试的第一行）: 1. Your first test text
[批量模式] 继续输入更多测试用例，输入空行或 'end' 结束:
2. Your second test text
3. Your third test text
[输入空行或 'end' 结束]
[显示批量测试结果]
```

### 2. 数据集测试接口

**启动方式：**
```bash
python skill_detection/test_rap_techniques.py --test-dataset
python skill_detection/test_rap_techniques.py --dataset
python skill_detection/test_rap_techniques.py -d
```

**功能：**
- 自动加载 `skill_detection/data/full_court_shot_correct.txt` 和 `skill_detection/data/full_court_shot_incorrect.txt`
- 测试所有正确和错误例子
- 输出详细的准确率、召回率、F1分数等指标
- 显示混淆矩阵和分数分布

**输出示例：**
```
======================================================================
Full-Court Shot 数据集测试
======================================================================

正确例子: 30 个
错误例子: 22 个
======================================================================

[测试正确例子]
----------------------------------------------------------------------
[OK] #1 : 100.0% - After that round the whole room watched your confi...
[OK] #2 : 100.0% - Mid battle your tough talk got repossessed I print...
...
[测试错误例子]
----------------------------------------------------------------------
[OK] #1 :   0.0% - After that round the room felt some kind of vibe I...
[FAIL] #2 : 100.0% - Mid battle the energy was different I clipped the ...
...
```

### 3. 命令行单条测试

**启动方式：**
```bash
python skill_detection/test_rap_techniques.py "你的文本"
```

**示例：**
```bash
python skill_detection/test_rap_techniques.py "After that round I pulled the clip titled it Where Your Voice Went and dropped the documentary"
```

### 4. 从文件批量测试

**Windows PowerShell:**
```powershell
Get-Content skill_detection/data/full_court_shot_correct.txt | python skill_detection/test_rap_techniques.py
```

**Linux/Mac:**
```bash
python skill_detection/test_rap_techniques.py < skill_detection/data/full_court_shot_correct.txt
```

## 接口独立性

两个接口完全独立：

- **`test_dataset()`** - 数据集测试接口
  - 独立的函数
  - 自动加载数据集文件
  - 输出评估指标

- **`run_interactive_mode()`** - 交互模式接口
  - 独立的函数
  - 支持单条和批量测试
  - 用户交互式输入

- **`main()`** - 统一入口
  - 根据命令行参数选择接口
  - 默认进入交互模式

## 命令行参数

| 参数 | 说明 |
|------|------|
| 无参数 | 默认进入交互模式 |
| `--interactive`, `-i` | 显式指定交互模式 |
| `--test-dataset`, `--dataset`, `-d` | 运行数据集测试 |
| `--help`, `-h` | 显示帮助信息 |
| `"文本"` | 命令行单条测试 |

## 代码结构

```
skill_detection/test_rap_techniques.py
├── format_technique_score()      # 格式化分数显示
├── test_text()                   # 单条测试
├── parse_batch_input()           # 解析批量输入
├── batch_test()                  # 批量测试
├── test_dataset()                # 数据集测试接口（独立）
├── run_interactive_mode()         # 交互模式接口（独立）
└── main()                        # 统一入口
```

## 常见问题

### Q: 如何运行数据集测试？
A: `python skill_detection/test_rap_techniques.py --test-dataset`

### Q: 如何进入交互模式？
A: 直接运行 `python skill_detection/test_rap_techniques.py` 或 `python skill_detection/test_rap_techniques.py --interactive`

### Q: 两个接口可以同时运行吗？
A: 不可以，每次只能运行一个接口。通过命令行参数选择。

### Q: 数据集测试会修改数据文件吗？
A: 不会，数据集测试是只读的，只读取数据文件进行测试。

