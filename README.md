# README
This project is the source code for the Paper：PHSGCL: Prompt-enhanced Hyperbolic Scattering Graph Contrastive Learning
This project employs a unified data loading interface. `hycrod/data.py -> load_dataset()`，  
**The training entry point remains unchanged.**，Uniformly run through the following methods：

```bash
python main.py ...
````

---

## 一. Environment and Dependency Installation

First, install the project's foundational dependencies：

```bash
pip install -r requirements.txt
```

To run OGB datasets (`ogbn-*`), you must additionally install OGB：

```bash
pip install ogb
```

---

## 二. Supported datasets (`--dataset`)

`--dataset` Parameters are **case-insensitive**; for example, `Cora / cora / CORA` will all function correctly.

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

## 三. Data Set Download and Storage Methods

### 1. PyTorch Geometric Built-in Datasets

（Planetoid / Amazon / Coauthor / WikiCS / WebKB）

* **No manual download required**
* Data is automatically downloaded by PyTorch Geometric during initial run
* Default cache path is the directory specified by `--data_root` (default `./data`)

example：

```bash
python main.py --dataset cora
```

The first run will automatically download and process the data; subsequent runs will directly use the cache.

---

### 2. OGB Dataset (`ogbn-*`)

* Requires prior installation:

  ```bash
  pip install ogb
  ```
* Data is automatically downloaded during the first run
* Cached in the directory specified by `--data_root`

example：

```bash
python main.py --dataset ogbn-arxiv --split ogb
```

---

## 四. Data splitting strategy (`--split`)

All split strategies will ultimately converge to generate:

* `data.train_mask`
* `data.val_mask`
* `data.test_mask`

To ensure consistency in the training code interface.

### 1. `public`

* Use the `train/val/test mask` included with the dataset
* Applicable to:

  * Planetoid
  * WikiCS
  * WebKB

example：

```bash
python main.py --dataset cora --split public
```

---

### 2. `random`

* Randomly partition node sets
* Supports ratio and random seed control for reproducibility
* Parameters:


  * `--train_ratio`
  * `--val_ratio`
  * `--test_ratio`
  * `--seed`
* Applicable to:

  * Amazon
  * Coauthor
  * and other datasets without standard public splits

example：

```bash
python main.py --dataset amazon-computers --split random \
  --train_ratio 0.1 --val_ratio 0.1 --test_ratio 0.8 --seed 42
```

---

### 3. `ogb`

* Use the official OGB split function (`get_idx_split()`)
* Automatically convert index splits to masks
* **Applies only to `ogbn-*` datasets**
* 
example：

```bash
python main.py --dataset ogbn-arxiv --split ogb
```

---

## 五. Minimum Viable Running Example

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

## 6. Frequently Asked Questions

### 1. Error when running `ogbn-*`: Missing OGB

Solution:

```bash
pip install ogb
```

### 2. Error using `--split public`: No built-in mask

Cause: This dataset does not provide a public split
Solution: Use `--split random` instead

### 3. Error: Invalid split ratio when using `--split random`

Solution: Ensure
`train_ratio + val_ratio + test_ratio = 1.0`
Example: `0.1 / 0.1 / 0.8`
