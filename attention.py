from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

PHOTO = Path(__file__).resolve().parent / "photo"
PHOTO.mkdir(exist_ok=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (seq_len, seq_len)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    output = weights @ V                      # (seq_len, d_v)
    return scores, weights, output

seq_len, d_k, d_v = 6, 8, 8
Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_v)

scores, weights, output = scaled_dot_product_attention(Q, K, V)

print("Attention Scores:\n", scores.round(3))
print("\nAttention Weights:\n", weights.round(3))
print("\nOutput:\n", output.round(3))

# 热力图：单头
plt.figure(figsize=(8, 3))
sns.heatmap(weights, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"K{i}" for i in range(seq_len)],
            yticklabels=[f"Q{i}" for i in range(seq_len)])
plt.title("Single-Head Attention Weights")
plt.xlabel("Key Position"); plt.ylabel("Query Position")
plt.tight_layout()
plt.savefig(PHOTO / "Scaled Dot-Product Attention_1.png", dpi=160)
plt.close()
print("\n已保存热力图：", PHOTO / "Scaled Dot-Product Attention_1.png")


def multi_head_attention(X, num_heads, d_model, d_k, d_v):
    """
    X: (seq_len, d_model)
    """
    seq_len = X.shape[0]
    W_Q = np.random.randn(d_model, num_heads * d_k) * 0.1
    W_K = np.random.randn(d_model, num_heads * d_k) * 0.1
    W_V = np.random.randn(d_model, num_heads * d_v) * 0.1
    W_O = np.random.randn(num_heads * d_v, d_model) * 0.1

    Q = X @ W_Q   # (seq_len, num_heads * d_k)
    K = X @ W_K
    V = X @ W_V

    # 拆分为多头
    Q = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)  # (h, seq_len, d_k)
    K = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, num_heads, d_v).transpose(1, 0, 2)

    # 每个头独立计算
    heads = []
    all_weights = []          # 顺手把各头权重留下来，供画热力图用
    for i in range(num_heads):
        sc = Q[i] @ K[i].T / np.sqrt(d_k)
        w = np.exp(sc) / np.exp(sc).sum(axis=-1, keepdims=True)
        all_weights.append(w)
        heads.append(w @ V[i])

    concat = np.concatenate(heads, axis=-1)   # (seq_len, h * d_v)
    output = concat @ W_O                       # (seq_len, d_model)
    return output, np.array(all_weights)        # (seq_len, d_model), (h, seq_len, seq_len)


X = np.random.randn(6, 64)
out, mh_weights = multi_head_attention(X, num_heads=8, d_model=64, d_k=8, d_v=8)
print("\nMulti-Head Output shape:", out.shape)  # (6, 64)

# 热力图：多头（每个头一张，横向排列）——各头关注模式不同，说明多头不是重复
num_heads = mh_weights.shape[0]
fig, axes = plt.subplots(1, num_heads, figsize=(3.1 * num_heads, 3.4))
for i, ax in enumerate(np.atleast_1d(axes)):
    sns.heatmap(mh_weights[i], annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0,
                cbar=(i == num_heads - 1),
                xticklabels=[f"K{j}" for j in range(seq_len)],
                yticklabels=[f"Q{j}" for j in range(seq_len)], ax=ax)
    ax.set_title(f"Head {i + 1}")
    ax.set_xlabel("Key Position")
    if i == 0:
        ax.set_ylabel("Query Position")
fig.suptitle("Multi-Head Attention Weights (8 heads)")
plt.tight_layout()
plt.savefig(PHOTO / "Scaled Dot-Product Attention_2.png", dpi=160)
plt.close()
print("已保存热力图：", PHOTO / "Scaled Dot-Product Attention_2.png")