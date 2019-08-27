# hard

1. softmax分类器的loss函数的反向传播梯度值求取
2. 协方差矩阵和svd分解

3. batchnorm和layernorm的区别
    batchnorm是在维度0上归一化
    layernorm是在维度1上归一化

    ```python
    # batchnorm
    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)
    # layernorm
    sample_mean = np.mean(x, axis=1, keepdims=True)
    sample_var = np.var(x, axis=1, keepdims=True)
    ```
