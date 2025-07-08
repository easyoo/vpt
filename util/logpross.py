import re
import os
import glob

def parse_log_file(log_path):
    """
    从指定路径的日志文件中解析出每个 epoch 的 loss 和评估指标。
    
    返回：
        results = {
            'bottle': {
                1: {'loss': 0.5681, 'I-Auroc': 0.9071, ...},
                2: {...},
                ...
            },
            ...
        }
    """
    # 正则表达式匹配 loss 行和评估指标表
    loss_pattern = re.compile(r"epoch \[(\d+)/(\d+)\], loss:(\d+\.\d+)")
    metric_table_header = re.compile(
         r"Class\s+I-Auroc\s+I-AP\s+I-F1\s+P-AUROC\s+P-AP\s+P-F1\s+P-AUPRO",re.IGNORECASE)
    metric_row_pattern = re.compile(
       r"^\s*(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$"
    )

    if not os.path.exists(log_path):
        raise FileNotFoundError(f"日志文件不存在：{log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    results = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 匹配 loss 行
        loss_match = loss_pattern.match(line)
        if loss_match:
            epoch = int(loss_match.group(1))

            # 检查下一行是否是指标表头
            if i + 1 < len(lines) and metric_table_header.match(lines[i + 1].strip()):
                # 再下一行是具体类别的数据行
                if i + 2 < len(lines):
                    metric_line = lines[i + 2].strip()
                    metric_match = metric_row_pattern.match(metric_line)
                    if metric_match:
                        cls = metric_match.group(1)

                        i_auroc = float(metric_match.group(2))
                        i_ap = float(metric_match.group(3))
                        i_f1 = float(metric_match.group(4))
                        p_auroc = float(metric_match.group(5))
                        p_ap = float(metric_match.group(6))
                        p_f1 = float(metric_match.group(7))
                        p_aupro = float(metric_match.group(8))

                        if cls not in results:
                            results[cls] = {}

                        results[cls][epoch] = {
                            "I-Auroc": i_auroc,
                            "I-AP": i_ap,
                            "I-F1": i_f1,
                            "P-AUROC": p_auroc,
                            "P-AP": p_ap,
                            "P-F1": p_f1,
                            "P-AUPRO": p_aupro,
                        }

                        i += 3  # 跳过处理的三行
                        continue
        i += 1

    return results



def result_cls_merge(dir):
    # 设置你想要搜索的目录路径
    directory = dir

    # 使用glob搜索所有后缀为.txt的文件
    search_pattern = os.path.join(directory, '**', '*.txt')
    found_files = glob.glob(search_pattern, recursive=True)
    items = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule','hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    results_tmp = {}
    results_ordered = {}
    # 输出找到的所有文件路径
    for file_path in found_files:
        result = parse_log_file(file_path)
        cls_key = list(result.keys())[0]
        results_tmp[cls_key] = result[cls_key]
    for item in items:
        results_ordered[item] = results_tmp[item]
    return results_ordered


def generate_markdown_table(results,epoch,store_dir):
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, "results.md")
    title = f"\n## Results for Epoch {epoch}\n"
    header = "| Class | I-Auroc | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |\n"
    header += "|-------|---------|------|------|---------|------|------|---------|\n"
    lines = [title,header]

    # 用于统计均值
    metrics_sum = {
        "I-Auroc": 0.0,
        "I-AP": 0.0,
        "I-F1": 0.0,
        "P-AUROC": 0.0,
        "P-AP": 0.0,
        "P-F1": 0.0,
        "P-AUPRO": 0.0,
    }
    count = 0

    for cls, epochs in results.items():
        if epoch not in epochs:
            raise ValueError(f"Epoch {epoch} not found for class {cls}.")
        metrics = epochs[epoch]
        line = f"| {cls} | {metrics['I-Auroc']*100:.2f} | {metrics['I-AP']*100:.2f} | {metrics['I-F1']*100:.2f} | {metrics['P-AUROC']*100:.2f} | {metrics['P-AP']*100:.2f} | {metrics['P-F1']*100:.2f} | {metrics['P-AUPRO']*100:.2f} |"
        lines.append(line + "\n")
        for k in metrics_sum:
            metrics_sum[k] += metrics[k]*100
        count += 1

    # 计算均值并添加到表格
    if count > 0:
        mean_line = f"| Mean |"
        for k in ["I-Auroc", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "P-AUPRO"]:
            mean_val = metrics_sum[k] / count
            mean_line += f" {mean_val:.2f} |"
        lines.append(mean_line + "\n")

    with open(path, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Markdown table saved to {path}")
    
    
if __name__ == "__main__":
    results = result_cls_merge("/home/jjquan/codebase/vpt/logs/evalute 2 shot")
    generate_markdown_table(results,70,"/home/jjquan/codebase/vpt/logs/evalute 2 shot")