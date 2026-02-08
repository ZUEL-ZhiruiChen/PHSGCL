"""
批量运行消融实验（多个数据集）
"""

import os
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_ablation_on_dataset(dataset, data_root, device, epochs, output_dir):
    """在单个数据集上运行完整消融实验"""
    print(f"\n{'='*80}")
    print(f"Running Ablation Study on: {dataset}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'ablation_study.py',
        '--dataset', dataset,
        '--data_root', data_root,
        '--device', device,
        '--epochs', str(epochs),
        '--output_dir', output_dir
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed on {dataset}: {e}")
        return False


def aggregate_results(datasets, output_dir):
    """汇总多个数据集的结果"""
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    all_results = []
    
    for dataset in datasets:
        csv_file = os.path.join(output_dir, f'ablation_results_{dataset}.csv')
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df['dataset'] = dataset
            all_results.append(df)
            print(f"✓ Loaded results for {dataset}")
        else:
            print(f"✗ Results not found for {dataset}")
    
    if not all_results:
        print("No results to aggregate!")
        return None
    
    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 保存汇总结果
    combined_path = os.path.join(output_dir, 'ablation_results_all_datasets.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"\n✓ Combined results saved to: {combined_path}")
    
    return combined_df


def plot_cross_dataset_comparison(combined_df, output_dir):
    """绘制跨数据集的对比图"""
    print("\n" + "="*80)
    print("GENERATING CROSS-DATASET VISUALIZATIONS")
    print("="*80)
    
    datasets = combined_df['dataset'].unique()
    variants = combined_df['variant'].unique()
    
    # 图1: 热力图 - 每个数据集上各变体的性能
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备数据矩阵
    pivot_data = combined_df.pivot_table(
        values='test_acc_linear',
        index='variant',
        columns='dataset',
        aggfunc='mean'
    )
    
    # 重新排序（完整模型在顶部）
    variant_order = ['full', 'no_prompt', 'no_hyp', 'no_scat', 'no_prompt_scat', 'baseline']
    pivot_data = pivot_data.reindex([v for v in variant_order if v in pivot_data.index])
    
    # 重命名
    variant_labels = {
        'full': 'PHSGCL',
        'no_prompt': 'w/o Prompt',
        'no_hyp': 'w/o Hyp',
        'no_scat': 'w/o Scat',
        'no_prompt_scat': 'w/o P+S',
        'baseline': 'Baseline'
    }
    pivot_data.index = [variant_labels.get(v, v) for v in pivot_data.index]
    
    # 绘制热力图
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                cbar_kws={'label': 'Test Accuracy'},
                linewidths=1, linecolor='white',
                vmin=pivot_data.min().min() * 0.95,
                vmax=pivot_data.max().max() * 1.01,
                ax=ax)
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Variant', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study - Performance Across Datasets', 
                 fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'cross_dataset_heatmap.pdf')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {heatmap_path}")
    plt.close()
    
    # 图2: 分组柱状图 - 每个数据集上的性能对比
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(datasets))
    width = 0.12
    
    colors = {
        'full': '#2E86AB',
        'no_prompt': '#A23B72',
        'no_hyp': '#F18F01',
        'no_scat': '#C73E1D',
        'no_prompt_scat': '#6A4C93',
        'baseline': '#90A959',
    }
    
    for i, variant in enumerate(variant_order):
        if variant not in variants:
            continue
        
        values = []
        for dataset in datasets:
            val = combined_df[
                (combined_df['dataset'] == dataset) & 
                (combined_df['variant'] == variant)
            ]['test_acc_linear'].values
            values.append(val[0] if len(val) > 0 else 0)
        
        offset = (i - len(variant_order)/2) * width
        ax.bar(x + offset, values, width, 
               label=variant_labels.get(variant, variant),
               color=colors.get(variant, '#888888'),
               alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('Ablation Study - Performance Comparison', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(fontsize=10, loc='lower left', ncol=2, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    barplot_path = os.path.join(output_dir, 'cross_dataset_barplot.pdf')
    plt.savefig(barplot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Bar plot saved to: {barplot_path}")
    plt.close()
    
    # 图3: 平均性能下降
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算每个变体相对于完整模型的平均性能下降
    avg_drops = []
    
    for variant in variant_order:
        if variant == 'full':
            continue
        
        drops = []
        for dataset in datasets:
            full_acc = combined_df[
                (combined_df['dataset'] == dataset) & 
                (combined_df['variant'] == 'full')
            ]['test_acc_linear'].values
            
            var_acc = combined_df[
                (combined_df['dataset'] == dataset) & 
                (combined_df['variant'] == variant)
            ]['test_acc_linear'].values
            
            if len(full_acc) > 0 and len(var_acc) > 0:
                drop = (full_acc[0] - var_acc[0]) / full_acc[0] * 100
                drops.append(drop)
        
        avg_drop = np.mean(drops) if drops else 0
        std_drop = np.std(drops) if drops else 0
        avg_drops.append((variant, avg_drop, std_drop))
    
    # 排序
    avg_drops.sort(key=lambda x: x[1], reverse=True)
    
    variants_plot = [variant_labels.get(v[0], v[0]) for v in avg_drops]
    means = [v[1] for v in avg_drops]
    stds = [v[2] for v in avg_drops]
    
    colors_plot = [colors.get(v[0], '#888888') for v in avg_drops]
    
    bars = ax.barh(variants_plot, means, xerr=stds, 
                   color=colors_plot, alpha=0.8,
                   edgecolor='black', linewidth=1.2,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' -{mean:.2f}% (±{std:.2f})',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Average Performance Drop (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model Variant', fontsize=13, fontweight='bold')
    ax.set_title('Average Performance Drop Across All Datasets', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.invert_yaxis()
    
    plt.tight_layout()
    avg_drop_path = os.path.join(output_dir, 'average_performance_drop.pdf')
    plt.savefig(avg_drop_path, dpi=300, bbox_inches='tight')
    print(f"✓ Average drop plot saved to: {avg_drop_path}")
    plt.close()


def create_summary_table(combined_df, output_dir):
    """创建汇总表格"""
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLE")
    print("="*80)
    
    datasets = combined_df['dataset'].unique()
    variants = combined_df['variant'].unique()
    
    # 计算每个变体在所有数据集上的平均性能
    summary_data = []
    
    variant_order = ['full', 'no_prompt', 'no_hyp', 'no_scat', 'no_prompt_scat', 'baseline']
    
    for variant in variant_order:
        if variant not in variants:
            continue
        
        var_data = combined_df[combined_df['variant'] == variant]
        
        avg_linear = var_data['test_acc_linear'].mean()
        std_linear = var_data['test_acc_linear'].std()
        avg_knn = var_data['test_acc_knn'].mean()
        avg_nmi = var_data['nmi_val'].mean()
        
        summary_data.append({
            'Variant': variant,
            'Avg Linear Probe': avg_linear,
            'Std Linear Probe': std_linear,
            'Avg k-NN': avg_knn,
            'Avg NMI': avg_nmi
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to: {summary_path}")
    
    # 打印到终端
    print("\n" + "="*80)
    print("SUMMARY: AVERAGE PERFORMANCE ACROSS ALL DATASETS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Batch Ablation Study')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['Cora', 'CiteSeer', 'PubMed'],
                        help='List of datasets')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                        help='Output directory')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only aggregate existing results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH ABLATION STUDY")
    print("="*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    if not args.skip_training:
        results = {}
        for dataset in args.datasets:
            success = run_ablation_on_dataset(
                dataset=dataset,
                data_root=args.data_root,
                device=args.device,
                epochs=args.epochs,
                output_dir=args.output_dir
            )
            results[dataset] = success
        
        # 打印训练总结
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        successful = [d for d, s in results.items() if s]
        failed = [d for d, s in results.items() if not s]
        
        print(f"✓ Successful: {len(successful)}/{len(args.datasets)}")
        if successful:
            print("  " + ", ".join(successful))
        
        if failed:
            print(f"✗ Failed: {len(failed)}/{len(args.datasets)}")
            print("  " + ", ".join(failed))
    
    # 汇总结果
    combined_df = aggregate_results(args.datasets, args.output_dir)
    
    if combined_df is not None:
        # 生成跨数据集对比图
        plot_cross_dataset_comparison(combined_df, args.output_dir)
        
        # 创建汇总表格
        create_summary_table(combined_df, args.output_dir)
    
    print("\n" + "="*80)
    print("BATCH ABLATION STUDY COMPLETED")
    print("="*80)
    print(f"All results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()