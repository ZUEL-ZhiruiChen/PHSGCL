"""
批量在多个数据集上运行 Rank Collapse 实验
"""

import os
import subprocess
import argparse


def run_experiment(dataset, data_root, device, epochs, output_dir):
    """运行单个数据集的实验"""
    print(f"\n{'='*80}")
    print(f"Running Rank Collapse Experiment on: {dataset}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', 'rank_collapse_experiment.py',
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
        print(f"✗ Failed to run experiment on {dataset}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch Rank Collapse Experiments')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs per model')
    parser.add_argument('--output_dir', type=str, default='./rank_collapse_results',
                        help='Output directory')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['Cora', 'CiteSeer', 'PubMed'],
                        help='List of datasets to run')
    
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH RANK COLLAPSE EXPERIMENTS")
    print("="*80)
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # 创建总输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行每个数据集
    results = {}
    for dataset in args.datasets:
        success = run_experiment(
            dataset=dataset,
            data_root=args.data_root,
            device=args.device,
            epochs=args.epochs,
            output_dir=args.output_dir
        )
        results[dataset] = success
    
    # 打印总结
    print("\n" + "="*80)
    print("BATCH EXPERIMENTS SUMMARY")
    print("="*80)
    
    successful = [d for d, s in results.items() if s]
    failed = [d for d, s in results.items() if not s]
    
    print(f"\n✓ Successful: {len(successful)}/{len(args.datasets)}")
    if successful:
        print("  " + ", ".join(successful))
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(args.datasets)}")
        print("  " + ", ".join(failed))
    
    print("\n" + "="*80)
    print(f"All results saved to: {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()