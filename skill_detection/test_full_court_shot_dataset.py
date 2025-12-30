#!/usr/bin/env python
"""
测试Full-Court Shot数据集的识别准确率
"""
import sys
from pathlib import Path

# 添加src目录到路径
src_dir = Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rap_techniques import detect_rap_techniques

def load_test_cases(file_path):
    """加载测试用例"""
    lines = file_path.read_text(encoding="utf-8").splitlines()
    test_cases = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 解析编号和文本
        import re
        match = re.match(r'^(\d+)\s+(.+)$', line)
        if match:
            test_id = match.group(1)
            text = match.group(2)
            test_cases.append((test_id, text))
    return test_cases

def main():
    root = Path(__file__).parent.parent
    
    # 加载正确和错误的例子
    correct_file = root / "data" / "full_court_shot_correct.txt"
    incorrect_file = root / "data" / "full_court_shot_incorrect.txt"
    
    if not correct_file.exists() or not incorrect_file.exists():
        print(f"错误: 找不到测试文件", file=sys.stderr)
        print(f"  正确例子: {correct_file}", file=sys.stderr)
        print(f"  错误例子: {incorrect_file}", file=sys.stderr)
        sys.exit(1)
    
    correct_cases = load_test_cases(correct_file)
    incorrect_cases = load_test_cases(incorrect_file)
    
    print("=" * 70)
    print("Full-Court Shot 数据集测试")
    print("=" * 70)
    print(f"\n正确例子: {len(correct_cases)} 个")
    print(f"错误例子: {len(incorrect_cases)} 个")
    print("=" * 70)
    
    # 测试正确例子
    print("\n[测试正确例子]")
    print("-" * 70)
    correct_detected = 0
    correct_scores = []
    
    for test_id, text in correct_cases:
        techniques_matrix = detect_rap_techniques([text])
        score = float(techniques_matrix.toarray()[0][0])  # Full-Court Shot分数
        
        correct_scores.append(score)
        if score > 0.3:  # 阈值
            correct_detected += 1
            status = "[OK]"
        else:
            status = "[FAIL]"
        
        print(f"{status} #{test_id:2s}: {score*100:5.1f}% - {text[:50]}...")
    
    # 测试错误例子
    print("\n[测试错误例子]")
    print("-" * 70)
    incorrect_rejected = 0
    incorrect_scores = []
    
    for test_id, text in incorrect_cases:
        techniques_matrix = detect_rap_techniques([text])
        score = float(techniques_matrix.toarray()[0][0])  # Full-Court Shot分数
        
        incorrect_scores.append(score)
        if score <= 0.3:  # 阈值
            incorrect_rejected += 1
            status = "[OK]"
        else:
            status = "[FAIL]"
        
        print(f"{status} #{test_id:2s}: {score*100:5.1f}% - {text[:50]}...")
    
    # 统计结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    true_positives = correct_detected
    false_negatives = len(correct_cases) - correct_detected
    true_negatives = incorrect_rejected
    false_positives = len(incorrect_cases) - incorrect_rejected
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (len(correct_cases) + len(incorrect_cases))
    
    print(f"\n正确例子识别率 (Recall): {recall*100:.1f}% ({correct_detected}/{len(correct_cases)})")
    print(f"错误例子拒绝率 (Specificity): {(true_negatives/len(incorrect_cases))*100:.1f}% ({incorrect_rejected}/{len(incorrect_cases)})")
    print(f"精确率 (Precision): {precision*100:.1f}%")
    print(f"F1分数: {f1:.3f}")
    print(f"总体准确率: {accuracy*100:.1f}%")
    
    print(f"\n混淆矩阵:")
    print(f"  True Positives (正确识别): {true_positives}")
    print(f"  False Negatives (漏检): {false_negatives}")
    print(f"  True Negatives (正确拒绝): {true_negatives}")
    print(f"  False Positives (误报): {false_positives}")
    
    # 分数分布
    if correct_scores:
        avg_correct = sum(correct_scores) / len(correct_scores)
        print(f"\n正确例子平均分数: {avg_correct*100:.1f}%")
    
    if incorrect_scores:
        avg_incorrect = sum(incorrect_scores) / len(incorrect_scores)
        print(f"错误例子平均分数: {avg_incorrect*100:.1f}%")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

