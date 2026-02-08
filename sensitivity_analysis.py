"""
参数敏感度分析 (Sensitivity Analysis)

系统地分析模型对关键超参数的敏感性：
1. 温度参数 τ (tau)
2. 跨视图对比权重 λ_cv (lambda_cv)
3. 正则化强度 γ (gamma)
4. 散射正则化权重 λ_scat (lambda_scat)
5. 双曲曲率 c (hyp_c)
6. 数据增强比例 (node_mask_ratio, edge_drop_ratio)
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from PHSGCL.models.hycrod import HyCroD, ModelConfig
from PHSGCL.models.prompting import PromptingConfig
from PHSGCL.trainer import TrainConfig, train
from PHSGCL.data import load_dataset
from PHSGCL.eval import linear_probe, knn_accuracy, nmi_score
from PHSGCL.utils.seed import set_seed


def run_single_experiment(
    param_name,
    param_value,
    dataset_name,
    data_root,
    device,
    base_config,
    epochs=200,
    seed=42,
    verbose=False
):
    """运行单次参数实验"""
    set_seed(seed)
    
    # 加载数据
    data, dataset, meta = load_dataset(
        name=dataset_name,
        root=data_root,
        normalize_features=True,
        split='public',
        seed=seed,
    )
    
    # 复制基础配置
    model_cfg_dict = {
        'in_dim': meta["num_features"],
        'gcn_hidden': base_config['gcn_hidden'],
        'gcn_out': base_config['gcn_out'],
        'hgcn_hidden': base_config['hgcn_hidden'],
        'hgcn_out': base_config['hgcn_out'],
        'gcn_layers': base_config['gcn_layers'],
        'hgcn_layers': base_config['hgcn_layers'],
        'proj_dim': base_config['proj_dim'],
        'pred_dim': base_config['pred_dim'],
        'dropout': base_config['dropout'],
        'tau': base_config['tau'],
        'hyp_c': base_config['hyp_c'],
        'prompting': PromptingConfig(enable=True),
    }
    
    train_cfg_dict = {
        'epochs': epochs,
        'lr': base_config['lr'],
        'weight_decay': base_config['weight_decay'],
        'tau': base_config['tau'],
        'lambda_cv': base_config['lambda_cv'],
        'gamma': base_config['gamma'],
    }
    
    aug_cfg_dict = {
        'node_mask_ratio': base_config['node_mask_ratio'],
        'edge_drop_ratio': base_config['edge_drop_ratio'],
    }
    
    reg_cfg_dict = {
        'lambda_scat': base_config['lambda_scat'],
        'lambda_1hop': base_config['lambda_1hop'],
        'lambda_mhop': base_config['lambda_mhop'],
        'k_hop': base_config['k_hop'],
    }
    
    # 更新目标参数
    if param_name in model_cfg_dict:
        model_cfg_dict[param_name] = param_value
    elif param_name in train_cfg_dict:
        train_cfg_dict[param_name] = param_value
    elif param_name in aug_cfg_dict:
        aug_cfg_dict[param_name] = param_value
    elif param_name in reg_cfg_dict:
        reg_cfg_dict[param_name] = param_value
    else:
        raise ValueError(f"Unknown parameter: {param_name}")
    
    # 同步 tau
    if param_name == 'tau':
        model_cfg_dict['tau'] = param_value
        train_cfg_dict['tau'] = param_value
    
    # 构建模型
    model_cfg = ModelConfig(**model_cfg_dict)
    model = HyCroD(model_cfg).to(device)
    
    # 构建训练配置
    train_cfg = TrainConfig(**train_cfg_dict)
    train_cfg.augment.node_mask_ratio = aug_cfg_dict['node_mask_ratio']
    train_cfg.augment.edge_drop_ratio = aug_cfg_dict['edge_drop_ratio']
    train_cfg.reg.lambda_scat = reg_cfg_dict['lambda_scat']
    train_cfg.reg.lambda_1hop = reg_cfg_dict['lambda_1hop']
    train_cfg.reg.lambda_mhop = reg_cfg_dict['lambda_mhop']
    train_cfg.reg.k_hop = reg_cfg_dict['k_hop']
    train_cfg.eval_interval = 50
    train_cfg.checkpoint_path = f"sensitivity_{param_name}_{param_value}_{dataset_name}.pt"
    
    if verbose:
        print(f"\n  Testing {param_name} = {param_value}")
    
    # 训练
    best_val_acc = train(model, data, device=device, cfg=train_cfg)
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        x = data.x.to(device)
        ei = data.edge_index.to(device)
        out = model.forward_views(x, ei, ei, x, seed=0)
        z = out["h_e1"]
    
    test_acc_linear = linear_probe(
        z, data.y.to(device),
        data.train_mask.to(device),
        data.test_mask.to(device),
        epochs=200, lr=0.01
    )
    
    val_acc_linear = linear_probe(
        z, data.y.to(device),
        data.train_mask.to(device),
        data.val_mask.to(device),
        epochs=200, lr=0.01
    )
    
    test_acc_knn = knn_accuracy(
        z, data.y.to(device),
        data.train_mask.to(device),
        data.test_mask.to(device),
        k=5
    )
    
    if verbose:
        print(f"    Val Acc: {val_acc_linear:.4f}, Test Acc: {test_acc_linear:.4f}")
    
    return {
        'param_name': param_name,
        'param_value': param_value,
        'val_acc': val_acc_linear,
        'test_acc': test_acc_linear,
        'test_acc_knn': test_acc_knn,
        'best_val_during_train': best_val_acc,
    }


def sensitivity_analysis(
    param_name,
    param_values,
    dataset_name,
    data_root,
    device,
    base_config,
    epochs=200,
    seed=42,
    output_dir='./sensitivity_results'
):
    """对单个参数进行敏感度分析"""
    print(f"\n{'='*70}")
    print(f"Sensitivity Analysis: {param_name}")
    print(f"{'='*70}")
    print(f"Testing values: {param_values}")
    
    results = []
    for value in tqdm(param_values, desc=f"Testing {param_name}"):
        result = run_single_experiment(
            param_name=param_name,
            param_value=value,
            dataset_name=dataset_name,
            data_root=data_root,
            device=device,
            base_config=base_config,
            epochs=epochs,
            seed=seed,
            verbose=True
        )
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'sensitivity_{param_name}_{dataset_name}.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    
    return results_df


def plot_sensitivity_curve(results_df, param_name, dataset_name, output_dir):
    """绘制敏感度曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: Test Accuracy vs Parameter
    ax1 = axes[0]
    x_values = results_df['param_value'].values
    test_acc = results_df['test_acc'].values
    val_acc = results_df['val_acc'].values
    
    ax1.plot(x_values, test_acc, 'o-', linewidth=2.5, markersize=8, 
             color='#2E86AB', label='Test Accuracy')
    ax1.plot(x_values, val_acc, 's--', linewidth=2, markersize=7, 
             color='#A23B72', label='Val Accuracy', alpha=0.8)
    
    # 标注最佳点
    best_idx = test_acc.argmax()
    best_val = x_values[best_idx]
    best_acc = test_acc[best_idx]
    ax1.plot(best_val, best_acc, '*', markersize=20, 
             color='#F18F01', label=f'Best: {best_val}', zorder=5)
    ax1.annotate(f'{best_acc:.4f}', 
                xy=(best_val, best_acc), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax1.set_xlabel(f'{param_name}', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title(f'(a) Accuracy vs {param_name} - {dataset_name}', 
                  fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # 如果是对数尺度参数，使用对数坐标
    if param_name in ['lr', 'weight_decay', 'hyp_c', 'gamma', 'lambda_scat']:
        ax1.set_xscale('log')
    
    # 子图2: Performance Stability (方差分析)
    ax2 = axes[1]
    
    # 计算相对于最佳性能的下降
    perf_drop = (best_acc - test_acc) / best_acc * 100
    
    colors = ['#2E86AB' if x == best_val else '#C73E1D' for x in x_values]
    bars = ax2.bar(range(len(x_values)), perf_drop, color=colors, 
                   alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for i, (bar, drop) in enumerate(zip(bars, perf_drop)):
        height = bar.get_height()
        if drop > 0.5:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{drop:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Parameter Value Index', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Performance Drop (%)', fontsize=13, fontweight='bold')
    ax2.set_title(f'(b) Performance Drop vs Best - {dataset_name}', 
                  fontsize=13, fontweight='bold', loc='left')
    ax2.set_xticks(range(len(x_values)))
    ax2.set_xticklabels([f'{v:.3g}' for v in x_values], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'sensitivity_{param_name}_{dataset_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")
    plt.close()


def plot_multi_param_comparison(all_results, dataset_name, output_dir):
    """绘制多参数对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    param_names = list(all_results.keys())
    sensitivities = []
    best_values = []
    
    for param_name in param_names:
        df = all_results[param_name]
        test_acc = df['test_acc'].values
        
        # 计算敏感度：最大值与最小值的差异
        sensitivity = (test_acc.max() - test_acc.min()) / test_acc.max() * 100
        sensitivities.append(sensitivity)
        
        # 最佳值
        best_idx = test_acc.argmax()
        best_val = df['param_value'].values[best_idx]
        best_values.append(f"{best_val:.3g}")
    
    # 排序
    sorted_indices = np.argsort(sensitivities)[::-1]
    sorted_params = [param_names[i] for i in sorted_indices]
    sorted_sens = [sensitivities[i] for i in sorted_indices]
    sorted_best = [best_values[i] for i in sorted_indices]
    
    # 颜色映射
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(sorted_params)))
    
    bars = ax.barh(sorted_params, sorted_sens, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加标签
    for i, (bar, sens, best) in enumerate(zip(bars, sorted_sens, sorted_best)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {sens:.2f}% (best: {best})',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Sensitivity (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=13, fontweight='bold')
    ax.set_title(f'Parameter Sensitivity Comparison - {dataset_name}', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'sensitivity_comparison_{dataset_name}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {save_path}")
    plt.close()


def create_sensitivity_table(all_results, dataset_name, output_dir):
    """创建敏感度分析汇总表格"""
    rows = []
    
    for param_name, df in all_results.items():
        test_acc = df['test_acc'].values
        param_values = df['param_value'].values
        
        best_idx = test_acc.argmax()
        worst_idx = test_acc.argmin()
        
        rows.append({
            'Parameter': param_name,
            'Best Value': f"{param_values[best_idx]:.4g}",
            'Best Acc': f"{test_acc[best_idx]:.4f}",
            'Worst Value': f"{param_values[worst_idx]:.4g}",
            'Worst Acc': f"{test_acc[worst_idx]:.4f}",
            'Range (%)': f"{(test_acc.max() - test_acc.min()) / test_acc.max() * 100:.2f}",
            'Std Dev': f"{test_acc.std():.4f}",
        })
    
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values('Range (%)', ascending=False)
    
    csv_path = os.path.join(output_dir, f'sensitivity_summary_{dataset_name}.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\n✓ Summary table saved to: {csv_path}")
    
    # 打印到终端
    print("\n" + "="*100)
    print(f"SENSITIVITY ANALYSIS SUMMARY - {dataset_name}")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(description='Parameter Sensitivity Analysis')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./sensitivity_results',
                        help='Output directory')
    parser.add_argument('--params', type=str, nargs='+',
                        default=['tau', 'lambda_cv', 'gamma', 'lambda_scat', 'hyp_c'],
                        help='Parameters to analyze')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # 基础配置
    base_config = {
        'gcn_hidden': 256,
        'gcn_out': 128,
        'hgcn_hidden': 256,
        'hgcn_out': 128,
        'gcn_layers': 2,
        'hgcn_layers': 2,
        'proj_dim': 128,
        'pred_dim': 128,
        'dropout': 0.2,
        'tau': 0.2,
        'hyp_c': 1.0,
        'lr': 1e-3,
        'weight_decay': 1e-5,
        'lambda_cv': 0.5,
        'gamma': 0.1,
        'node_mask_ratio': 0.3,
        'edge_drop_ratio': 0.3,
        'lambda_scat': 1.0,
        'lambda_1hop': 1.0,
        'lambda_mhop': 1.0,
        'k_hop': 3,
    }
    
    # 参数搜索空间
    param_ranges = {
        'tau': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        'lambda_cv': [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
        'gamma': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        'lambda_scat': [0.0, 0.5, 1.0, 2.0, 5.0],
        'hyp_c': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'node_mask_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'edge_drop_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'lr': [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
        'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    }
    
    print("="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Parameters to analyze: {', '.join(args.params)}")
    print("="*70)
    
    all_results = {}
    
    # 对每个参数进行敏感度分析
    for param in args.params:
        if param not in param_ranges:
            print(f"\n✗ Warning: No predefined range for parameter '{param}', skipping...")
            continue
        
        results_df = sensitivity_analysis(
            param_name=param,
            param_values=param_ranges[param],
            dataset_name=args.dataset,
            data_root=args.data_root,
            device=device,
            base_config=base_config,
            epochs=args.epochs,
            seed=args.seed,
            output_dir=args.output_dir
        )
        
        all_results[param] = results_df
        
        # 绘制单参数曲线
        plot_sensitivity_curve(results_df, param, args.dataset, args.output_dir)
    
    # 绘制多参数对比图
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("GENERATING COMPARISON PLOTS")
        print("="*70)
        plot_multi_param_comparison(all_results, args.dataset, args.output_dir)
        create_sensitivity_table(all_results, args.dataset, args.output_dir)
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETED")
    print("="*70)
    print(f"All results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()