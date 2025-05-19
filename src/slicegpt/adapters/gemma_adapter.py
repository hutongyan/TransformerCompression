import torch
from torch import FloatTensor, Tensor
from torch.nn import Linear, Module
from transformers import PretrainedConfig, PreTrainedTokenizerBase
# 确保导入 GemmaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaConfig, GemmaDecoderLayer, GemmaForCausalLM, GemmaRMSNorm

from slicegpt.model_adapter import LayerAdapter, ModelAdapter


class CompressedGemmaDecoderLayer(GemmaDecoderLayer):
    """
    Gemma的压缩层实现，类似于Llama的实现，增加了shortcut_Q以支持残差连接的旋转。
    """

    # 父类 GemmaDecoderLayer 的 __init__ 签名是 (self, config: GemmaConfig, layer_idx: int)
    # 我们需要确保自定义属性被初始化
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        #self.attn_shortcut_Q = None
        #self.mlp_shortcut_Q = None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None, # 显式添加 position_ids
        past_key_value: tuple[Tensor, Tensor] | None = None, # HuggingFace Gemma 使用 tuple[Tensor, Tensor]
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        cache_position: Tensor | None = None, # 可能需要在Gemma中使用
        # **kwargs, # 移除非预期的kwargs，或者显式处理它们
    ) -> tuple: # 返回类型可以更精确，例如 tuple[Tensor, Optional[Tensor], Optional[tuple[Tensor, Tensor]]]
        """
        与原始GemmaDecoderLayer的forward方法基本一致，但增加了对shortcut_Q的处理。
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        #print(f"DEBUG: Layer {self.layer_idx} forward: hidden_states dtype after input_layernorm: {hidden_states.dtype}")
        #print(f"DEBUG: Layer {self.layer_idx} forward: self.self_attn.q_proj.weight dtype: {self.self_attn.q_proj.weight.dtype}")

        # Self Attention
        # 修改: self.attention -> self.self_attn
        # 修改: 传递 position_ids
        # GemmaAttention的forward不接受任意kwargs，移除**kwargs传递
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # GemmaAttention 返回 (attn_output, attn_weights, past_key_value)
        hidden_states = attention_outputs[0]
        self_attn_weights = attention_outputs[1] # attn_weights (None if output_attentions=False)
        present_key_value = attention_outputs[2] # past_key_value (None if use_cache=False)


        if self.attn_shortcut_Q is not None:
            rotated_residual = torch.matmul(residual, self.attn_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.mlp_shortcut_Q is not None:
            rotated_residual = torch.matmul(residual, self.mlp_shortcut_Q)
            hidden_states = rotated_residual + hidden_states
        else:
            hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class GemmaLayerAdapter(LayerAdapter):
    """
    适配Gemma模型的层结构。
    """

    def __init__(self, layer: GemmaDecoderLayer) -> None:
        super().__init__()
        self._layer: GemmaDecoderLayer = layer

    @property
    def layer(self) -> Module:
        return self._layer

    @property
    def hidden_states_args_position(self) -> int:
        return 0

    @property
    def hidden_states_output_position(self) -> int:
        return 0

    def get_first_layernorm(self) -> Module:
        return self.layer.input_layernorm

    def get_second_layernorm(self) -> Module:
        return self.layer.post_attention_layernorm

    def get_attention_inputs(self) -> list[Linear]:
        # 修改: self.layer.attention -> self.layer.self_attn
        return [self.layer.self_attn.q_proj, self.layer.self_attn.k_proj, self.layer.self_attn.v_proj]

    def get_attention_output(self) -> Linear:
        # 修改: self.layer.attention.o_proj -> self.layer.self_attn.o_proj
        return self.layer.self_attn.o_proj

    def get_mlp_inputs(self) -> list[Linear]:
        return [self.layer.mlp.gate_proj, self.layer.mlp.up_proj]

    def get_mlp_output(self) -> Linear:
        return self.layer.mlp.down_proj


class GemmaModelAdapter(ModelAdapter):
    """
    适配Gemma模型的整体结构。
    """

    def __init__(self, model: GemmaForCausalLM) -> None:
        super().__init__()
        self._model: GemmaForCausalLM = model

    @property
    def model(self) -> Module:
        return self._model

    @property
    def config(self) -> PretrainedConfig:
        return self._model.config

    @property
    def config_type(self) -> type:
        return GemmaConfig

    @property
    def parallel_blocks(self) -> bool:
        return False

    @property
    def seqlen(self) -> int:
        return self.config.max_position_embeddings

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    @property
    def should_bake_mean_into_linear(self) -> bool:
        # 修改: Gemma使用RMSNorm，没有可学习的均值(beta参数)，所以是False
        return False

    @property
    def original_layer_type(self) -> type:
        return GemmaDecoderLayer

    @property
    def original_layer_norm_type(self) -> type:
        # 修改: Gemma使用GemmaRMSNorm
        return GemmaRMSNorm

    @property
    def layer_adapter_type(self) -> type:
        return GemmaLayerAdapter

    @property
    def compressed_layer_type(self) -> type:
        return CompressedGemmaDecoderLayer

    @property
    def use_cache(self) -> bool:
        # 确保config中有use_cache属性，如果transformers版本较旧可能没有
        return getattr(self.config, "use_cache", False)


    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.config.use_cache = value

    def compute_output_logits(self, input_ids: Tensor, **kwargs) -> FloatTensor:
        # GemmaForCausalLM.forward 接受 attention_mask, position_ids 等参数
        # 确保这些参数在kwargs中或者从dataloader中正确提供
        return self.model(input_ids=input_ids, **kwargs).logits

    def convert_layer_to_compressed(self, layer: Module, layer_idx: int | None) -> Module:
        if layer_idx is None:
            # GemmaDecoderLayer 的 __init__ 需要 layer_idx
            raise ValueError("layer_idx cannot be None for Gemma when converting to compressed layer.")
        print(f"DEBUG: convert_layer_to_compressed (layer {layer_idx}): self.config.torch_dtype = {self.config.torch_dtype}")
        # compressed_layer_type 是 CompressedGemmaDecoderLayer，其父类构造函数需要 config 和 layer_idx
        compressed_layer = self.compressed_layer_type(self.config, layer_idx=layer_idx).to(self.config.torch_dtype)
        print(f"DEBUG: convert_layer_to_compressed (layer {layer_idx}): q_proj dtype after init = {compressed_layer.self_attn.q_proj.weight.dtype}")
        compressed_layer.load_state_dict(layer.state_dict(), strict=True)
        print(f"DEBUG: convert_layer_to_compressed (layer {layer_idx}): q_proj dtype after load_state_dict = {compressed_layer.self_attn.q_proj.weight.dtype}")
        return compressed_layer

    def get_layers(self) -> list[LayerAdapter]:
        return [self.layer_adapter_type(layer) for layer in self.model.model.layers]

    def get_raw_layer_at(self, index: int) -> Module:
        return self.model.model.layers[index]

    def set_raw_layer_at(self, index: int, new_layer: Module) -> None:
        self.model.model.layers[index] = new_layer

    def get_embeddings(self) -> list[Module]:
        return [self.model.model.embed_tokens]

    def get_pre_head_layernorm(self) -> Module | None: # 返回类型可以为 None
        return self.model.model.norm

    def get_lm_head(self) -> Linear:
        return self.model.lm_head

    @classmethod
    def _from_pretrained(
        cls,
        model_name: str, # model_name 在此实现中未使用
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> "GemmaModelAdapter": # 更具体的返回类型
        model = GemmaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            token=token,
            local_files_only=local_files_only,
            trust_remote_code=True # Gemma模型通常需要
        )
        print(f"DEBUG: _from_pretrained: model.config.torch_dtype = {model.config.torch_dtype}")
        print(f"DEBUG: _from_pretrained: specified dtype arg = {dtype}")
        # 关键修复：确保模型配置对象的torch_dtype与期望的dtype一致
        if model.config.torch_dtype != dtype:
            print(f"INFO: Overriding model.config.torch_dtype from {model.config.torch_dtype} to {dtype}")
            model.config.torch_dtype = dtype

        print(f"DEBUG: _from_pretrained: model.config.torch_dtype AFTER override = {model.config.torch_dtype}")
        # model.config.torch_dtype = dtype # from_pretrained 应该已经处理了torch_dtype
        return GemmaModelAdapter(model)

    @classmethod
    def _from_uninitialized(
        cls,
        model_name: str, # model_name 在此实现中未使用
        model_path: str,
        *,
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> "GemmaModelAdapter": # 更具体的返回类型
        class UninitializedGemmaForCausalLM(GemmaForCausalLM):
            def _init_weights(self, module: Module) -> None: # Hugging Face _init_weights 签名需要一个module参数
                # Prevent weight initialization
                pass

        config = GemmaConfig.from_pretrained(
            model_path,
            # torch_dtype=dtype, # Config的from_pretrained通常不直接接受torch_dtype
            token=token,
            local_files_only=local_files_only,
            trust_remote_code=True # Gemma模型通常需要
        )
        # 在config上设置torch_dtype，如果模型实例化时需要
        print(f"DEBUG: _from_uninitialized: config.torch_dtype BEFORE override = {getattr(config, 'torch_dtype', 'N/A')}")
        print(f"DEBUG: _from_uninitialized: specified dtype arg (target) = {dtype}")

        # 关键修复：确保配置对象的torch_dtype与期望的dtype一致
        # 这会影响后续基于此config创建模型时的参数初始化类型
        if getattr(config, 'torch_dtype', None) != dtype:
             print(f"INFO: Overriding config.torch_dtype to {dtype}")
             config.torch_dtype = dtype

        print(f"DEBUG: _from_uninitialized: config.torch_dtype AFTER override = {config.torch_dtype}")

        model = UninitializedGemmaForCausalLM(config)
        # .to(dtype) 应该在 .to(device) 之后或者在模块级别完成，这里通常不需要
        # model = model.to(dtype=dtype)
        return GemmaModelAdapter(model)

    def post_init(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # Gemma tokenizer 需要设置 pad_token
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                # print(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
            elif tokenizer.bos_token_id is not None: # 有些模型可能只有 bos_token
                tokenizer.pad_token_id = tokenizer.bos_token_id
                # print(f"Set pad_token_id to bos_token_id: {tokenizer.bos_token_id}")
            else: # 如果都没有，则添加一个新的pad_token
                # print("Adding new pad_token [PAD]")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # 确保config中的pad_token_id与tokenizer一致
        if self.config.pad_token_id is None or self.config.pad_token_id != tokenizer.pad_token_id :
             self.config.pad_token_id = tokenizer.pad_token_id

        # 确保config中有use_cache属性
        if not hasattr(self.config, 'use_cache'):
            self.config.use_cache = False # 如果不存在，默认为False