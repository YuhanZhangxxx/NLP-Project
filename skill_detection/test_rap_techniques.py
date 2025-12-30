#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
说唱技巧检测测试工具
直接输入句子测试说唱技巧识别功能
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

# 检查依赖
try:
    import scipy
except ImportError:
    print("错误: 缺少依赖 scipy", file=sys.stderr)
    print("请运行: pip install scipy", file=sys.stderr)
    sys.exit(1)

# 添加src目录到路径
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from rap_techniques import detect_rap_techniques, detect_victim_state_by_pos, NLTK_AVAILABLE, VICTIM_STATE_KEYWORDS, VICTIM_STATE_PATTERNS
    import re
except ImportError as e:
    print(f"错误: 无法导入 rap_techniques 模块: {e}", file=sys.stderr)
    sys.exit(1)

def format_technique_score(score):
    """格式化技巧分数显示"""
    percentage = score * 100
    if score >= 0.7:
        return f"[!!] {percentage:.1f}% (强烈)"
    elif score >= 0.5:
        return f"[*]  {percentage:.1f}% (明显)"
    elif score >= 0.3:
        return f"[+]  {percentage:.1f}% (存在)"
    else:
        return f"     {percentage:.1f}% (微弱)"

def check_victim_state_method1(text):
    """方法1：关键词库识别击溃语义"""
    s_lower = text.lower()
    
    # 检查关键词
    for kw in VICTIM_STATE_KEYWORDS:
        if ' ' in kw:
            if re.search(r'\b' + re.escape(kw) + r'\b', s_lower):
                return True, f"关键词: '{kw}'"
        else:
            if kw in s_lower:
                return True, f"关键词: '{kw}'"
    
    # 检查模式
    for pattern in VICTIM_STATE_PATTERNS:
        if r'was\s+(different|off|strange)' in pattern:
            continue
        if re.search(pattern, s_lower):
            return True, f"模式匹配"
    
    return False, "未识别"

def test_text(text, show_methods=False):
    """测试文本的说唱技巧"""
    print("=" * 70)
    print("说唱技巧检测结果")
    print("=" * 70)
    print(f"\n输入文本:")
    print(f'"{text}"')
    
    # 如果启用方法显示，显示两种击溃语义识别方法的结果
    if show_methods:
        print("\n" + "-" * 70)
        print("击溃语义识别方法对比:")
        print("-" * 70)
        
        # 方法1：关键词库识别
        method1_result, method1_detail = check_victim_state_method1(text)
        method1_status = "✓ 识别到" if method1_result else "✗ 未识别"
        print(f"  方法1 (关键词库识别): {method1_status}")
        if method1_result:
            print(f"    └─ {method1_detail}")
        
        # 方法2：词性识别
        if NLTK_AVAILABLE:
            method2_result = detect_victim_state_by_pos(text)
            method2_status = "✓ 识别到" if method2_result else "✗ 未识别"
            print(f"  方法2 (词性识别): {method2_status}")
        else:
            method2_result = False
            print(f"  方法2 (词性识别): NLTK不可用")
        
        # 最终结果（OR逻辑）
        final_victim_state = method1_result or method2_result
        final_status = "✓ 有击溃语义" if final_victim_state else "✗ 无击溃语义"
        print(f"  最终结果 (方法1 OR 方法2): {final_status}")
    
    print("\n" + "-" * 70)
    
    # 检测技巧
    techniques_matrix = detect_rap_techniques([text])
    techniques_array = techniques_matrix.toarray()[0]
    
    techniques = {
        "Full-Court Shot (+5.0分)": float(techniques_array[0]),
        "Slam Dunk (+4.25分)": float(techniques_array[1]),
        "Half-Court Shot (+3.75分)": float(techniques_array[2]),
        "Alley-Oop/Assist (+3.5分)": float(techniques_array[3])
    }
    
    print("\n检测到的技巧:")
    print("-" * 70)
    
    # 按分数排序显示
    sorted_techniques = sorted(techniques.items(), key=lambda x: x[1], reverse=True)
    
    for name, score in sorted_techniques:
        formatted_score = format_technique_score(score)
        print(f"  {name:30s} : {formatted_score}")
    
    # 显示最高分的技巧
    max_tech = max(techniques.items(), key=lambda x: x[1])
    if max_tech[1] > 0.3:
        print("\n" + "-" * 70)
        print(f">>> 主要技巧: {max_tech[0]} ({max_tech[1]*100:.1f}%)")
    
    print("=" * 70)
    print()

def parse_batch_input(text: str):
    """
    解析批量测试输入
    识别以数字开头的行作为编号，后面的内容作为测试文本
    支持格式：
    1. 文本内容
    1. 文本内容
    2 文本内容
    3.文本内容
    """
    lines = text.splitlines()
    test_cases = []
    current_id = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            # 空行，如果当前有文本，保存它
            if current_text:
                test_cases.append((current_id, " ".join(current_text)))
                current_text = []
            continue
        
        # 检查是否以数字开头（可能带点或空格）
        import re
        match = re.match(r'^(\d+)[.\s]*(.*)$', line)
        if match:
            # 如果之前有文本，先保存
            if current_text:
                test_cases.append((current_id, " ".join(current_text)))
                current_text = []
            
            # 新的编号和文本
            current_id = match.group(1)
            text_part = match.group(2).strip()
            if text_part:
                current_text.append(text_part)
        else:
            # 不是以数字开头，作为当前文本的延续
            if current_id is not None:
                current_text.append(line)
            else:
                # 没有编号，作为独立测试用例
                test_cases.append((None, line))
    
    # 保存最后一个测试用例
    if current_text:
        test_cases.append((current_id, " ".join(current_text)))
    
    return test_cases

def batch_test(texts_with_ids, show_methods=False):
    """批量测试多个文本"""
    print("=" * 70)
    print("批量说唱技巧检测")
    print("=" * 70)
    print(f"\n共 {len(texts_with_ids)} 个测试用例\n")
    
    results = []
    for idx, (test_id, text) in enumerate(texts_with_ids, 1):
        if not text.strip():
            continue
        
        print(f"\n{'='*70}")
        print(f"测试用例 #{idx}" + (f" (编号: {test_id})" if test_id else ""))
        print(f"{'='*70}")
        
        # 如果启用方法对比显示，显示击溃语义识别方法对比
        if show_methods:
            print(f"\n文本: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
            print("\n" + "-" * 70)
            print("击溃语义识别方法对比:")
            print("-" * 70)
            
            # 方法1：关键词库识别
            method1_result, method1_detail = check_victim_state_method1(text)
            method1_status = "✓ 识别到" if method1_result else "✗ 未识别"
            print(f"  方法1 (关键词库识别): {method1_status}")
            if method1_result:
                print(f"    └─ {method1_detail}")
            
            # 方法2：词性识别
            if NLTK_AVAILABLE:
                method2_result = detect_victim_state_by_pos(text)
                method2_status = "✓ 识别到" if method2_result else "✗ 未识别"
                print(f"  方法2 (词性识别): {method2_status}")
            else:
                method2_result = False
                print(f"  方法2 (词性识别): NLTK不可用")
            
            # 最终结果（OR逻辑）
            final_victim_state = method1_result or method2_result
            final_status = "✓ 有击溃语义" if final_victim_state else "✗ 无击溃语义"
            print(f"  最终结果 (方法1 OR 方法2): {final_status}")
        
        # 检测技巧
        techniques_matrix = detect_rap_techniques([text])
        techniques_array = techniques_matrix.toarray()[0]
        
        techniques = {
            "Full-Court Shot (+5.0分)": float(techniques_array[0]),
            "Slam Dunk (+4.25分)": float(techniques_array[1]),
            "Half-Court Shot (+3.75分)": float(techniques_array[2]),
            "Alley-Oop/Assist (+3.5分)": float(techniques_array[3])
        }
        
        if not show_methods:
            print(f"\n文本: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        print("\n检测结果:")
        print("-" * 70)
        
        # 按分数排序显示
        sorted_techniques = sorted(techniques.items(), key=lambda x: x[1], reverse=True)
        
        for name, score in sorted_techniques:
            formatted_score = format_technique_score(score)
            print(f"  {name:30s} : {formatted_score}")
        
        # 显示最高分的技巧
        max_tech = max(techniques.items(), key=lambda x: x[1])
        if max_tech[1] > 0.3:
            print("\n" + "-" * 70)
            print(f">>> 主要技巧: {max_tech[0]} ({max_tech[1]*100:.1f}%)")
        
        results.append({
            'id': test_id,
            'text': text,
            'techniques': techniques,
            'max_technique': max_tech[0] if max_tech[1] > 0.3 else None,
            'max_score': max_tech[1]
        })
    
    # 汇总统计
    print(f"\n\n{'='*70}")
    print("批量测试汇总")
    print(f"{'='*70}")
    
    technique_counts = {
        "Full-Court Shot": 0,
        "Slam Dunk": 0,
        "Half-Court Shot": 0,
        "Alley-Oop/Assist": 0,
        "无显著技巧": 0
    }
    
    for result in results:
        if result['max_technique']:
            tech_name = result['max_technique'].split(' (')[0]  # 去掉分数部分
            if tech_name in technique_counts:
                technique_counts[tech_name] += 1
        else:
            technique_counts["无显著技巧"] += 1
    
    print("\n主要技巧分布:")
    for tech, count in technique_counts.items():
        if count > 0:
            percentage = count / len(results) * 100
            print(f"  {tech:25s}: {count:3d} 个 ({percentage:5.1f}%)")
    
    print(f"\n总计: {len(results)} 个测试用例")
    print("=" * 70)
    
    return results

# ============================================================================
# 数据集测试接口（独立）
# ============================================================================

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

def load_csv_test_cases(csv_file):
    """从CSV文件加载测试用例"""
    import csv
    test_cases = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_id = row.get('id', '')
            text = row.get('text', '')
            label = row.get('label', '')
            if text:
                test_cases.append((test_id, text, label))
    return test_cases

def test_dataset():
    """测试Full-Court Shot数据集的识别准确率（独立接口）"""
    root = Path(__file__).parent  # 数据文件在同一目录的data子目录
    
    # 优先尝试从CSV文件加载
    csv_file = root / "data" / "full_court_shot_test.csv"
    if csv_file.exists():
        print("从CSV文件加载测试用例...")
        all_cases = load_csv_test_cases(csv_file)
        correct_cases = [(tid, text) for tid, text, label in all_cases if label == 'correct']
        incorrect_cases = [(tid, text) for tid, text, label in all_cases if label == 'incorrect']
    else:
        # 回退到txt文件
        correct_file = root / "data" / "full_court_shot_correct.txt"
        incorrect_file = root / "data" / "full_court_shot_incorrect.txt"
        
        if not correct_file.exists() or not incorrect_file.exists():
            print(f"错误: 找不到测试文件", file=sys.stderr)
            print(f"  CSV文件: {csv_file}", file=sys.stderr)
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

def test_dataset_from_csv(csv_file):
    """从CSV文件测试数据集（独立接口）"""
    all_cases = load_csv_test_cases(csv_file)
    
    correct_cases = [(tid, text) for tid, text, label in all_cases if label == 'correct']
    incorrect_cases = [(tid, text) for tid, text, label in all_cases if label == 'incorrect']
    
    print("=" * 70)
    print("Full-Court Shot 数据集测试 (CSV)")
    print("=" * 70)
    print(f"\n数据源: {csv_file}")
    print(f"正确例子: {len(correct_cases)} 个")
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

# ============================================================================
# 交互模式接口（独立）
# ============================================================================

def run_interactive_mode(show_methods=False):
    """运行交互模式（独立接口）"""
    print("=" * 70)
    print("说唱技巧检测测试工具 - 交互模式")
    print("=" * 70)
    print("\n[使用说明]:")
    print("  - 单条测试: 直接输入文本，按回车")
    print("  - 批量测试: 输入多行，每行以数字开头（如: 1. 文本）")
    print("  - 显示方法对比: 输入 '--methods' 或 '-m' 开启方法对比显示")
    print("  - 退出: 输入 'quit' 或 'exit'")
    print("  - 批量模式结束: 输入空行或 'end'")
    if show_methods:
        print("\n[当前模式]: 已启用方法对比显示")
    print("=" * 70)
    print()
    
    while True:
        try:
            # 读取第一行
            first_line = input("请输入文本（或批量测试的第一行）: ").strip()
            
            if not first_line:
                continue
            
            if first_line.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            # 切换方法显示模式
            if first_line.lower() in ['--methods', '-m', 'methods']:
                show_methods = not show_methods
                print(f"\n[方法对比显示]: {'已启用' if show_methods else '已关闭'}\n")
                continue
            
            # 检查是否是批量模式（以数字开头）
            if re.match(r'^\d+[.\s]', first_line):
                # 批量模式
                print("\n[批量模式] 继续输入更多测试用例，输入空行或 'end' 结束:")
                print("-" * 70)
                
                lines = [first_line]
                while True:
                    try:
                        line = input().strip()
                        if not line or line.lower() == 'end':
                            break
                        lines.append(line)
                    except (EOFError, KeyboardInterrupt):
                        break
                
                if lines:
                    input_text = "\n".join(lines)
                    test_cases = parse_batch_input(input_text)
                    if test_cases:
                        batch_test(test_cases, show_methods=show_methods)
                    else:
                        print("错误: 没有解析到测试用例", file=sys.stderr)
                print()
            else:
                # 单条测试模式
                test_text(first_line, show_methods=show_methods)
                print()
                
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except EOFError:
            print("\n\n再见!")
            break

# ============================================================================
# 主函数 - 统一入口，接口独立
# ============================================================================

def main():
    """主函数 - 统一入口，接口独立"""
    # 检查帮助
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("=" * 70)
        print("说唱技巧检测测试工具 - 使用说明")
        print("=" * 70)
        print("\n[交互模式] (默认):")
        print("  python skill_detection/test_rap_techniques.py")
        print("  python skill_detection/test_rap_techniques.py --interactive")
        print("  python skill_detection/test_rap_techniques.py -i")
        print("  python skill_detection/test_rap_techniques.py --interactive --methods  # 启用方法对比")
        print("  - 直接输入文本，按回车测试单条")
        print("  - 输入多行（以数字开头）自动进入批量模式")
        print("  - 输入 '--methods' 或 '-m' 切换方法对比显示")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("\n[数据集测试]:")
        print("  python skill_detection/test_rap_techniques.py --test-dataset")
        print("  python skill_detection/test_rap_techniques.py --dataset")
        print("  python skill_detection/test_rap_techniques.py -d")
        print("  - 测试Full-Court Shot数据集的识别准确率（自动从CSV或txt文件加载）")
        print("\n[CSV文件测试]:")
        print("  python skill_detection/test_rap_techniques.py --test-csv")
        print("  python skill_detection/test_rap_techniques.py --csv")
        print("  - 从CSV文件测试数据集")
        print("\n[命令行单条测试]:")
        print("  python skill_detection/test_rap_techniques.py \"你的文本\"")
        print("\n[从文件批量测试]:")
        print("  Get-Content skill_detection/data/full_court_shot_test.csv | python skill_detection/test_rap_techniques.py")
        print("  或: python skill_detection/test_rap_techniques.py < skill_detection/data/full_court_shot_test.csv")
        print("\n[批量格式]:")
        print("  1. Your first test text")
        print("  2. Your second test text")
        print("  3 Your third test text (不带点也可以)")
        sys.exit(0)
    
    # 检查数据集测试模式
    if len(sys.argv) > 1 and sys.argv[1] in ['--test-dataset', '--dataset', '-d']:
        test_dataset()
        return
    
    # 检查CSV文件测试模式
    if len(sys.argv) > 1 and sys.argv[1] in ['--test-csv', '--csv']:
        csv_file = Path(__file__).parent / "data" / "full_court_shot_test.csv"
        if not csv_file.exists():
            print(f"错误: 找不到CSV文件: {csv_file}", file=sys.stderr)
            sys.exit(1)
        test_dataset_from_csv(csv_file)
        return
    
    # 检查交互模式（显式指定）
    if len(sys.argv) > 1 and sys.argv[1] in ['--interactive', '-i']:
        show_methods = '--methods' in sys.argv or '-m' in sys.argv
        run_interactive_mode(show_methods=show_methods)
        return
    
    # 命令行参数：单条测试（非模式参数）
    if len(sys.argv) > 1 and sys.argv[1] not in ['--interactive', '-i', '--test-dataset', '--dataset', '-d', '--methods', '-m']:
        # 过滤掉 --methods 参数
        args = [arg for arg in sys.argv[1:] if arg not in ['--methods', '-m']]
        text = " ".join(args)
        show_methods = '--methods' in sys.argv or '-m' in sys.argv
        test_text(text, show_methods=show_methods)
        return
    
    # 从stdin读取（管道输入）
    if not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
        if not input_text:
            print("错误: 没有输入文本", file=sys.stderr)
            sys.exit(1)
        
        # 尝试解析为批量测试
        test_cases = parse_batch_input(input_text)
        if len(test_cases) > 1 or (len(test_cases) == 1 and test_cases[0][0] is not None):
            # 多个测试用例或带编号，使用批量模式
            batch_test(test_cases)
        else:
            # 单个测试用例
            test_text(test_cases[0][1] if test_cases else input_text)
        return
    
    # 默认：交互模式
    show_methods = '--methods' in sys.argv or '-m' in sys.argv
    run_interactive_mode(show_methods=show_methods)

if __name__ == "__main__":
    main()

