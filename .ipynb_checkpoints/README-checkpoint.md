
# 数据集与运行说明（README）

本项目采用统一的数据加载接口 `hycrod/data.py -> load_dataset()`，  
**训练入口保持不变**，统一通过以下方式运行：

```bash
python main.py ...
````

---

## 一、环境与依赖安装

首先安装项目基础依赖：

```bash
pip install -r requirements.txt
```

如果需要运行 OGB 数据集（`ogbn-*`），需额外安装 OGB：

```bash
pip install ogb
```

---

## 二、支持的数据集（`--dataset`）

`--dataset` 参数**大小写不敏感**，例如 `Cora / cora / CORA` 均可正常运行。

### 1. Planetoid

* `citeseer`
* `pubmed`

### 2. Amazon

* `amazon-computers`
* `amazon-photos`

### 3. Coauthor

* `coauthor-cs`
* `coauthor-physics`

### 4. WikiCS

* `wikics`

### 5. WebKB

* `webkb-cornell`
* `webkb-texas`
* `webkb-wisconsin`

### 6. OGB Node Classification

* `ogbn-arxiv`
* `ogbn-products`

---

## 三、数据集下载与存放方式

### 1. PyTorch Geometric 内置数据集

（Planetoid / Amazon / Coauthor / WikiCS / WebKB）

* **无需手动下载**
* 数据在首次运行时由 PyTorch Geometric 自动下载
* 默认缓存路径为 `--data_root` 指定目录（默认 `./data`）

示例：

```bash
python main.py --dataset cora
```

首次运行会自动下载并处理数据，之后将直接使用缓存。

---

### 2. OGB 数据集（`ogbn-*`）

* 需先安装：

  ```bash
  pip install ogb
  ```
* 数据在首次运行时自动下载
* 同样缓存到 `--data_root` 指定目录下

示例：

```bash
python main.py --dataset ogbn-arxiv --split ogb
```

---

## 四、数据划分策略（`--split`）

所有 split 策略最终都会统一生成：

* `data.train_mask`
* `data.val_mask`
* `data.test_mask`

以保证训练代码接口一致。

### 1. `public`

* 使用数据集自带的 `train/val/test mask`
* 适用于：

  * Planetoid
  * WikiCS
  * WebKB

示例：

```bash
python main.py --dataset cora --split public
```

---

### 2. `random`

* 随机划分节点集合
* 支持比例与随机种子控制，可复现
* 参数：

  * `--train_ratio`
  * `--val_ratio`
  * `--test_ratio`
  * `--seed`
* 适用于：

  * Amazon
  * Coauthor
  * 以及其他无标准 public split 的数据集

示例：

```bash
python main.py --dataset amazon-computers --split random \
  --train_ratio 0.1 --val_ratio 0.1 --test_ratio 0.8 --seed 42
```

---

### 3. `ogb`

* 使用 OGB 官方提供的 split（`get_idx_split()`）
* 自动将 index split 转换为 mask
* **仅适用于 `ogbn-*` 数据集**

示例：

```bash
python main.py --dataset ogbn-arxiv --split ogb
```

---

## 五、最小可用运行示例

```bash
# Cora（public split）
python main.py --dataset cora --split public

# Amazon Computers（random split）
python main.py --dataset amazon-computers --split random \
  --train_ratio 0.1 --val_ratio 0.1 --test_ratio 0.8 --seed 42

# Coauthor CS（random split）
python main.py --dataset coauthor-cs --split random \
  --train_ratio 0.1 --val_ratio 0.1 --test_ratio 0.8 --seed 42

# WebKB Cornell（public split）
python main.py --dataset webkb-cornell --split public

# OGBN-Arxiv（需先安装 ogb）
python main.py --dataset ogbn-arxiv --split ogb
```

---

## 六、常见问题

### 1. 运行 `ogbn-*` 报错提示缺少 OGB

解决：

```bash
pip install ogb
```

### 2. 使用 `--split public` 报错没有内置 mask

原因：该数据集不提供 public split
解决：改用 `--split random`

### 3. 使用 `--split random` 报错比例不合法

解决：确保
`train_ratio + val_ratio + test_ratio = 1.0`
例如：`0.1 / 0.1 / 0.8`
