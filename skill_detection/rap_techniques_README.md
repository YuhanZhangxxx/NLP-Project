# 说唱技巧识别模块使用指南

## 模块结构

`rap_techniques.py` 模块包含所有说唱技巧的检测逻辑。每个技巧都有独立的检测函数，方便维护和扩展。

## 当前支持的技巧

1. **Full-Court Shot** (+5.0分) - `detect_full_court_shot()`
2. **Slam Dunk** (+4.25分) - `detect_slam_dunk()`
3. **Half-Court Shot** (+3.75分) - `detect_half_court_shot()`
4. **Alley-Oop/Assist** (+3.5分) - `detect_alley_oop()`

## 如何添加新技巧

### 步骤 1: 定义关键词和模式

在模块顶部添加新技巧的关键词集合和正则模式：

```python
NEW_TECHNIQUE_KEYWORDS = {
    'keyword1', 'keyword2', 'keyword3',
    # ... 更多关键词
}

NEW_TECHNIQUE_PATTERNS = [
    r'\bpattern1\s+\w+',  # 模式1
    r'\bpattern2\s+\w+',  # 模式2
    # ... 更多模式
]
```

### 步骤 2: 创建检测函数

添加检测函数，返回 0-1 的分数：

```python
def detect_new_technique(text: str) -> float:
    """
    检测新技巧
    描述：这个技巧的特点是什么
    """
    s_lower = text.lower()
    score = 0.0
    
    # 关键词匹配
    keyword_count = sum(1 for kw in NEW_TECHNIQUE_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.15, 0.4)  # 调整权重
    
    # 模式匹配
    pattern_matches = sum(1 for pattern in NEW_TECHNIQUE_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.2, 0.4)  # 调整权重
    
    # 自定义逻辑
    if re.search(r'\bspecial\s+pattern', s_lower):
        score += 0.2
    
    return min(score, 1.0)
```

### 步骤 3: 更新主函数

在 `detect_rap_techniques()` 函数中添加新技巧的调用：

```python
def detect_rap_techniques(texts):
    rows = []
    for text in texts:
        s = text if isinstance(text, str) else ""
        
        # 现有技巧
        full_court_score = detect_full_court_shot(s)
        slam_dunk_score = detect_slam_dunk(s)
        half_court_score = detect_half_court_shot(s)
        alley_oop_score = detect_alley_oop(s)
        
        # 新技巧
        new_technique_score = detect_new_technique(s)
        
        # 添加到结果中（注意顺序）
        rows.append([
            full_court_score, 
            slam_dunk_score, 
            half_court_score, 
            alley_oop_score,
            new_technique_score  # 新技巧
        ])
    
    A = np.asarray(rows, dtype=float)
    return sparse.csr_matrix(A)
```

### 步骤 4: 更新技巧信息

在 `TECHNIQUE_INFO` 字典中添加新技巧的信息：

```python
TECHNIQUE_INFO = {
    # ... 现有技巧
    "new_technique": {
        "name": "New Technique",
        "points": "+X.X Points",
        "description": "技巧描述"
    }
}
```

### 步骤 5: 更新相关文件

1. **`src/train_lr_plus.py`**: 更新配置中的技巧列表
2. **`src/analyze_song.py`**: 更新 `analyze_rap_techniques()` 函数中的技巧映射
3. **`src/test_rap_techniques.py`**: 更新测试工具中的技巧列表

## 技巧检测函数模板

```python
def detect_your_technique(text: str) -> float:
    """
    检测 [技巧名称] 技巧
    
    参数:
        text: 输入文本
    
    返回:
        float: 技巧强度分数 (0-1)
    """
    s_lower = text.lower()
    score = 0.0
    
    # 1. 关键词匹配
    keyword_count = sum(1 for kw in YOUR_KEYWORDS if kw in s_lower)
    score += min(keyword_count * 0.15, 0.4)  # 调整权重和上限
    
    # 2. 模式匹配
    pattern_matches = sum(1 for pattern in YOUR_PATTERNS if re.search(pattern, s_lower))
    score += min(pattern_matches * 0.2, 0.4)  # 调整权重和上限
    
    # 3. 特殊逻辑检测
    if re.search(r'\bspecial_pattern', s_lower):
        score += 0.2
    
    # 4. 组合特征检测
    if condition1 and condition2:
        score += 0.3
    
    return min(score, 1.0)  # 确保不超过1.0
```

## 测试新技巧

使用测试工具验证新技巧：

```bash
python src/test_rap_techniques.py "你的测试文本"
```

## 注意事项

1. **分数范围**: 所有检测函数必须返回 0-1 之间的浮点数
2. **性能**: 避免使用过于复杂的正则表达式，保持检测速度
3. **权重调整**: 根据实际效果调整关键词和模式的权重
4. **向后兼容**: 添加新技巧时，确保现有代码仍然可以工作（通过 `feats_extra.py` 的导入）

## 示例：添加新技巧的完整流程

假设要添加 "Triple Threat" (+4.0分) 技巧：

1. 在模块顶部定义关键词和模式
2. 创建 `detect_triple_threat()` 函数
3. 在 `detect_rap_techniques()` 中调用新函数
4. 在 `TECHNIQUE_INFO` 中添加信息
5. 更新相关文件中的技巧列表
6. 测试验证

