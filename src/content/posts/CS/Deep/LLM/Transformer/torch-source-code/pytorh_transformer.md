---
title: "PyTorch 中的 Transformer 源码 [LLM]"
published: 2026-05-27 10:00:00
category: CS
image: "./cover.png"
---

上节我们自己实现了一个 `Transformer`，这节我们通过源码来看一下 `PyTorch` 官方是如何实现 `Transformer` 的。内容按以下顺序进行：从最简单的注意力组件开始向上直到顶层的 `Transformer` 类。

## 1. 多头自注意力的实现

`MHA` 在 `pytorch/torch/nn/modules/activation.py` 中

```python
class MultiheadAttention(Module):

	# batch_first 在模块生命周期内不会改变，可以作为一个常量处理
	__constants__ = ["batch_first"]
	# 类型注解：bias_k,v 可以是 Tensor 或 None
    bias_k: torch.Tensor | None
    bias_v: torch.Tensor | None

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
		# 创建一个字典参数，后续通过 **factory_kwargs 直接解包
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
		# 检查 qkv 的维度是否一样
        self._qkv_same_embed_dim = (self.kdim == embed_dim and self.vdim == embed_dim)

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
		# 划分子空间
        self.head_dim = embed_dim // num_heads

		# qkvo 初始化
        if not self._qkv_same_embed_dim:
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

		# qkv 在投影时是否需要 bias (Q = W_q @ X + b_q...)
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

		# 与普通的 Linear 层相同，只是名称带有非量化的显式标记
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

		# 是否在 K 和 V 序列的开头添加一个额外的可学习向量
        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()
```

## 2. LN 的实现

### 2.1 LayerNorm 类

`LN` 在 `pytorch/torch/nn/modules/normalization.py` 中:

```python

_shape_t = int | list[int] | Size

class LayerNorm(Module):

	def __init__(
        self,
		# 定义要归一化的维度的大小
        normalized_shape: _shape_t,
		# 公式中的 ε
        eps: float = 1e-5,
		# 使用 γ 和 β
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
		# 给成员赋值
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
			# γ 初始值是 1
            init.ones_(self.weight)
            if self.bias is not None:
				# β 初始值是 0
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
		# 将参数传给 functional.layer_norm()
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps
        )


```

### 2.2 functional.py

`pytorch/torch/nn/functional.py` 将底层的高效 `C++` 实现包装成优雅的 `Python API`，供 `nn.Module` 和开发者调用

```python
def layer_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor | None = None,
    bias: Tensor | None = None,
    eps: float = 1e-5,
) -> Tensor:
    r"""Apply Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
	# 如果自定义了张量 (variadic: 可变参数)，将调用交给自定义类的 __torch_function__ 方法
    if has_torch_function_variadic(input, weight, bias):
        return handle_torch_function(
            layer_norm,
            (input, weight, bias),
            input,
            normalized_shape,
            weight=weight,
            bias=bias,
            eps=eps,
        )
	# 否则调用 C++ 实现的高性能版本
    return torch.layer_norm(
        input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled
    )
```

### 2.3 c++ 实现

`LN` 计算底层在 `pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu` 中实现，`aten` 文件夹是 `PyTorch` 的核心底层张量计算库，存放所有基础张量操作的 `C++/CUDA` 实现

```cpp
std::tuple<Tensor, Tensor, Tensor> layer_norm_cuda(
    const Tensor& input,
    IntArrayRef normalized_shape,
    const std::optional<Tensor>& weight_opt /* optional */,
    const std::optional<Tensor>& bias_opt /* optional */,
    double eps) {

  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
 // 解引用获取 Tensor 引用
  const Tensor& weight = *weight_maybe_owned;
  
  c10::MaybeOwned<Tensor> bias_maybe_owned =
      at::borrow_from_optional_tensor(bias_opt);
  const Tensor& bias = *bias_maybe_owned;

  // auto 让编译器自动推断变量的类型
  auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
  // M_N 的 第一个和第二个元素
  auto M = M_N.first;
  auto N = M_N.second;
  auto X = input.expect_contiguous();
  auto gamma = weight.expect_contiguous();
  auto beta = bias.expect_contiguous();

  // 输出张量 Y
  Tensor Y = at::native::empty_like(...);
  // 均值张量和 std 张量
  Tensor mean = at::empty(...);
  Tensor rstd = at::empty(...);
  // M > 0 时调用 kernel
  if (M > 0) {
    LayerNormKernelImpl(*X, *gamma, *beta, M, N, eps, &Y, &mean, &rstd);
  }
  ...
  // 计算得到的均值和 std 通过 view 改变维度与输入对齐进行广播
  mean = mean.view(stat_shape);
  rstd = rstd.view(stat_shape);
  // 返回 mean, std 用于 bp
  return std::make_tuple(std::move(Y), std::move(mean), std::move(rstd));
}

void LayerNormKernelImpl(...) {
  // 根据输入张量的数据类型，自动生成并调用针对该类型的模板代码
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "LayerNormKernelImpl",
      [&]() {
        using acc_t = acc_type<scalar_t, true>;
		// 执行计算
        LayerNormKernelImplInternal<scalar_t, acc_t>(
            X, gamma, beta, M, N, static_cast<acc_t>(eps), Y, mean, rstd);
      });
}

// 最后在 LayerNormKernelImplInternal 中调用 RowwiseMomentsCUDAKernel 和 LayerNormForwardCUDAKernel 计算
```

## 3. encoder 的实现

`encoder`，`decoder` 都在 `pytorch/torch/nn/modules/transformer.py` 中

### 3.1 TransformerEncoderLayer 类

```python
class TransformerEncoderLayer(Module):

	def __init__(
        self,
        d_model: int,
        nhead: int,
		# FFN 的维度
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
		# 激活函数
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
		# Pre-LN 还是 Post-LN
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
		factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
		# 自注意力
        self.self_attn = MultiheadAttention(...)
		
		# FFN
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

		# LN
        self.norm_first = norm_first
		# norm1 用于自注意力，norm2 用于 FFN
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
		# ... (略去激活函数处理细节)
		# 定义激活函数 
		self.activation = activation
		
	def forward(
        self,
        src: Tensor,
		# 注意力掩码
        src_mask: Tensor | None = None,
		# pad 掩码
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
		# 将传入的 mask 格式规范化
		src_key_padding_mask = F._canonical_mask(...)
		src_mask = F._canonical_mask(...)
		# 计算逻辑
		x = src
		# Pre-LN
        if self.norm_first:
			# 注意力计算
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
			# Post-LN
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
			# FFN
            x = self.norm2(x + self._ff_block(x))

        return x

	def _sa_block(...) -> Tensor:
		# 上面定义了 self.self_attn = MultiheadAttention(...)
        x = self.self_attn(
			# 自注意力 qkv 都是 x
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

	def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
```

### 3.2 TransformerEncoder 类

通过 `TransformerEncoder` 类对 `TransformerEncoderLayer` 进行堆叠:

```python
class TransformerEncoder(Module):
	def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Module | None = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
		super().__init__()
		# 创建 num_layers 个编码器层的深拷贝
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

	def forward(
        self,
        src: Tensor,
        mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool | None = None,
    ) -> Tensor:
	
	seq_len = _get_seq_len(src, batch_first)
	
	for mod in self.layers:
			# 循环调用 model 的 forward 方法
            output = mod(
                output,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
            )

	return output
```

## 4. decoder 的实现

`TransformerDecoder` 堆叠 `layer` 的方式与 `TransformerEncoder` 类似，略去:

```python
class TransformerDecoderLayer(Module):

	def __init__(
        self,
        d_model: int,
        nhead: int,
		# FFN 的中间维度
        dim_feedforward: int = 2048,
        ...
    ) -> None:
		self.self_attn = MultiheadAttention(...)
	    self.multihead_attn = MultiheadAttention(...)

	# 定义 FFN 和 残差
	# 跟 encoder 相比多一个交叉注意力，因此需要 norm3 和 dropout3

	def forward(
        self,
		# decoder 的输入 编码序列
        tgt: Tensor,
		# encoder 输出
        memory: Tensor,
		# 屏蔽未来 token
        tgt_mask: Tensor | None = None,
		# 用于交叉注意力
        memory_mask: Tensor | None = None,
		# 屏蔽目标序列的 <pad>
        tgt_key_padding_mask: Tensor | None = None,
		# 屏蔽源序列 <pad>
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
	
		x = tgt
        if self.norm_first:
			# mmha
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            )
			# 交叉注意力
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

		def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
			# kv 是 encoder 输出
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

```

## 5. 组件整合

`Transformer` 类是整个模型的顶层组装器，它负责将编码器和解码器组合成一个完整的端到端模型：

```python
class Transformer(Module):

	def __init__(
        self,
		# 默认值
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.relu,
        custom_encoder: Any | None = None,
        custom_decoder: Any | None = None,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:

		# 搭建网络
		if custom_encoder is not None:
		else:
            encoder_layer = TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            encoder_norm = LayerNorm(
                d_model,
                eps=layer_norm_eps,
                bias=bias,
                **factory_kwargs,
            )
			# 创建 encoder
            self.encoder = TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm
            )

		if custom_decoder is not None:
		else:
            decoder_layer = TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,
            )
            decoder_norm = LayerNorm(
                d_model,
                eps=layer_norm_eps,
                bias=bias,
                **factory_kwargs,
            )
			# 创建 decoder
            self.decoder = TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm
            )

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

	def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        src_is_causal: bool | None = None,
        tgt_is_causal: bool | None = None,
        memory_is_causal: bool = False,
    ) -> Tensor:

		# encoder 输出, 调用 encoder 的 forward()
		memory = self.encoder(
            src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
        )

		# decoder 输出
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal,
            memory_is_causal=memory_is_causal,
        )
        return output
```

