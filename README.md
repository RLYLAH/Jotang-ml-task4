# 《Attention Is All You Need》

论文：*Attention Is All You Need*（Vaswani et al., NeurIPS 2017，[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)）


---

## 一、文件

| 文件 | 内容 |
|---|---|
| [`问题解答.md`](问题解答.md) | 任务要求的 8 个问题，逐题作答 |
| [`阅读笔记.md`](阅读笔记.md) | 用到的资料与工具、读不懂的地方与理解过程、仍存在的疑问 |
| [`拓展_模型迁移.md`](拓展_模型迁移.md) | 拓展三（ViT）、拓展四（Encoder-only / Decoder-only） |
| [`attention.py`](attention.py) | 拓展一 + 拓展二：单头与多头注意力，并输出两张热力图 |
| [`photo/Scaled Dot-Product Attention_1.png`](photo/Scaled%20Dot-Product%20Attention_1.png) | 拓展一：单头注意力权重热力图 |
| [`photo/Scaled Dot-Product Attention_2.png`](photo/Scaled%20Dot-Product%20Attention_2.png) | 拓展二：8 个头的注意力权重热力图 |
| [`photo/transformer.jpg`](photo/transformer.jpg) | 手绘的 Transformer 结构草图（Encoder / Decoder 分列左右） |

---

## 二、Figure 1 拆解：Encoder 与 Decoder

原图 Figure 1 是左右两半，**左半是 Encoder、右半是 Decoder**，两者都由 $N=6$ 个相同层堆叠。
手绘的版本见 [`photo/transformer.jpg`](photo/transformer.jpg)。

---

## 三、代码与结果

`attention.py` 跑拓展一和拓展二：

- `scaled_dot_product_attention(Q, K, V)`：$\text{softmax}(QK^{\top}/\sqrt{d_k})V$
- `multi_head_attention(X, num_heads, d_model, d_k, d_v)`：随机投影 → 拆成 8 个头 → 各头独立注意力 → 拼接 → $W^O$

规模很小（序列长 6，$d_{model}=64$，8 个头、每头 8 维），输出：

1. 单头注意力的 **scores / weights / output** 三个矩阵，打印到终端；
2. [`photo/Scaled Dot-Product Attention_1.png`](photo/Scaled%20Dot-Product%20Attention_1.png)：单头权重热力图；
3. [`photo/Scaled Dot-Product Attention_2.png`](photo/Scaled%20Dot-Product%20Attention_2.png)：8 个头各自一张热力图，横向排列。

**两张图已经生成并放在 `photo/` 里，可直接查看。
8 个头的高权重格子落在不同位置，即**多头不是把同一件事重复多次


---
