"""
批量运行参数敏感度分析（多个数据集）
"""

import os
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def run_sensitivity_on_dataset(dataset, data_root, device, epochs, output_dir, params):
    """在单个数据集上运行敏感度分析"""
    print(f"\n{'='*80}")
    print(f"Running Sensitivity Analysis on: {dataset}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'sensitivity_analysis.py',
        '--dataset', dataset,
        '--data_root', data_root,
        '--device', device,
        '--epochs', str(epochs),
        '--output_dir', output_dir,
        '--params'
    ] + params
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed on {dataset}: {e}")
        return False


def plot_cross_dataset_sensitivity(datasets, param_name, output_dir):
    """绘制跨数据集的敏感度对比"""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
    
    if len(datasets) == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        csv_file = os.path.join(output_dir, f'sensitivity_{param_name}_{dataset}.csv')
        
        if not os.path.exists(csv_file):
            print(f"✗ Results not found for {dataset}")
            continue
        
        df = pd.read_csv(csv_file)
        ax = axes[idx]
        
        x_values = df['param_value'].values
        test_acc = df['test_acc'].values
        
        ax.plot(x_values, test_acc, 'o-', linewidth=2.5, markersize=8, color='#2E86AB')
        
        # 标注最佳点
        best_idx = test_acc.argmax()
        best_val = x_values[best_idx]
        best_acc = test_acc[best_idx]
        ax.plot(best_val, best_acc, '*', markersize=18, color='#F18F01', zorder=5)
        ax.annotate(f'{best_acc:.4f}', 
                    xy=(best_val, best_acc), 
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
        
        ax.set_xlabel(f'{param_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 对数尺度
        if param_name in ['lr', 'weight_decay', 'hyp_c', 'gamma', 'lambda_scat']:
            ax.set_xscale('log')
    
    plt.suptitle(f'Sensitivity Analysis: {param_name} Across Datasets', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'cross_dataset_sensitivity_{param_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Cross-dataset plot saved to: {save_path}")
    plt.close()


def create_cross_dataset_summary(datasets, params, output_dir):
    """创建跨数据集的敏感度汇总表"""
    print("\n" + "="*80)
    print("CREATING CROSS-DATASET SUMMARY")
    print("="*80)
    
    summary_data = []
    
    for param in params:
        for dataset in datasets:
            csv_file = os.path.join(output_dir, f'sensitivity_{param}_{dataset}.csv')
            
            if not os.path.exists(csv_file):
                continue
            
            df = pd.read_csv(csv_file)
            test_acc = df['test_acc'].values
            param_values = df['param_value'].values
            
            best_idx = test_acc.argmax()
            
            summary_data.append({
                'Dataset': dataset,
                'Parameter': param,
                'Best Value': param_values[best_idx],
                'Best Acc': test_acc[best_idx],
                'Sensitivity (%)': (test_acc.max() - test_acc.min()) / test_acc.max() * 100,
                'Std Dev': test_acc.std(),
            })
    
    if not summary_data:
        print("No data to summarize!")
        return None
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存
    csv_path = os.path.join(output_dir, 'cross_dataset_sensitivity_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"✓ Cross-dataset summary saved to: {csv_path}")
    
    # 打印
    print("\n" + "="*100)
    print("CROSS-DATASET SENSITIVITY SUMMARY")
    print("="*100)
    for param in params:
        param_data = summary_df[summary_df['Parameter'] == param]
        if not param_data.empty:
            print(f"\n{param}:")
            print(param_data.to_string(index=False))
    print("="*100)
    
    return summary_df


def plot_sensitivity_heatmap(datasets, params, output_dir):
    """绘制敏感度热力图"""
    print("\n" + "="*80)
    print("GENERATING SENSITIVITY HEATMAP")
    print("="*80)
    
    # 准备数据矩阵
    sensitivity_matrix = []
    
    for param in params:
        row = []
        for dataset in datasets:
            csv_file = os.path.join(output_dir, f'sensitivity_{param}_{dataset}.csv')
            
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                test_acc = df['test_acc'].values
                sensitivity = (test_acc.max() - test_acc.min()) / test_acc.max() * 100
                row.append(sensitivity)
            else:
                row.append(np.nan)
        
        sensitivity_matrix.append(row)
    
    sensitivity_matrix = np.array(sensitivity_matrix)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(sensitivity_matrix, 
                annot=True, fmt='.2f', 
                cmap='YlOrRd',
                xticklabels=datasets,
                yticklabels=params,
                cbar_kws={'label': 'Sensitivity (%)'},
                linewidths=1, linecolor='white',
                ax=ax)
    
    ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=13, fontweight='bold')
    ax.set_title('Parameter Sensitivity Across Datasets', 
                 fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sensitivity_heatmap.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Batch Sensitivity Analysis')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['Cora', 'CiteSeer', 'PubMed'],
                        help='List of datasets')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    parser.add_argument('--output_dir', type=str, default='./sensitivity_results',
                        help='Output directory')
    parser.add_argument('--params', type=str, nargs='+',
                        default=['tau', 'lambda_cv', 'gamma', 'lambda_scat', 'hyp_c'],
                        help='Parameters to analyze')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only aggregate existing results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH SENSITIVITY ANALYSIS")
    print("="*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Parameters: {', '.join(args.params)}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("="*80)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    if not args.skip_training:
        results = {}
        for dataset in args.datasets:
            success = run_sensitivity_on_dataset(
                dataset=dataset,
                data_root=args.data_root,
                device=args.device,
                epochs=args.epochs,
                output_dir=args.output_dir,
                params=args.params
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
    
    # 生成跨数据集可视化
    print("\n" + "="*80)
    print("GENERATING CROSS-DATASET VISUALIZATIONS")
    print("="*80)
    
    for param in args.params:
        plot_cross_dataset_sensitivity(args.datasets, param, args.output_dir)
    
    # 创建汇总
    create_cross_dataset_summary(args.datasets, args.params, args.output_dir)
    
    # 绘制热力图
    plot_sensitivity_heatmap(args.datasets, args.params, args.output_dir)
    
    print("\n" + "="*80)
    print("BATCH SENSITIVITY ANALYSIS COMPLETED")
    print("="*80)
    print(f"All results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()