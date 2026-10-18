"""Weight loading and decode API for Qwen3.5-0.8B bf16 megakernel."""

import os
import struct
import torch

NUM_LAYERS = 24
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3584
VOCAB_SIZE = 248320
MAX_SEQ_LEN = 2048


def _half_dtype():
    """Return the 16-bit dtype matching the compiled kernel: bf16 on SM>=80, fp16 otherwise."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            return torch.float16
    return torch.bfloat16

FA_NUM_Q_HEADS = 8
FA_NUM_KV_HEADS = 2
FA_HEAD_DIM = 256
FA_Q_SIZE = FA_NUM_Q_HEADS * FA_HEAD_DIM
FA_QPROJ_SIZE = FA_Q_SIZE * 2
FA_KV_SIZE = FA_NUM_KV_HEADS * FA_HEAD_DIM

DN_NUM_HEADS = 16
DN_KEY_DIM = 128
DN_VALUE_DIM = 128
DN_QK_SIZE = DN_NUM_HEADS * DN_KEY_DIM
DN_V_SIZE = DN_NUM_HEADS * DN_VALUE_DIM
DN_CONV_CHANNELS = DN_QK_SIZE * 2 + DN_V_SIZE
DN_CONV_KERNEL = 4

LAYER_TYPE = [0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1]

_decode = None
_decode_many = None
_decode_batch = None
_prefill = None
_max_safe_decode_blocks = None
_set_decode_blocks = None
_compiled_config = None


def _load_op():
    global _decode, _decode_many, _decode_batch, _prefill, _max_safe_decode_blocks, _set_decode_blocks, _compiled_config
    if _decode is None:
        import qwen35_megakernel_bf16_C
        _decode = torch.ops.qwen35_megakernel_bf16_C.decode
        _decode_many = torch.ops.qwen35_megakernel_bf16_C.decode_many
        _decode_batch = torch.ops.qwen35_megakernel_bf16_C.decode_batch
        _prefill = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16
        _max_safe_decode_blocks = torch.ops.qwen35_megakernel_bf16_C.max_safe_decode_blocks
        _set_decode_blocks = torch.ops.qwen35_megakernel_bf16_C.set_decode_blocks
        ns = torch.ops.qwen35_megakernel_bf16_C
        _compiled_config = {
            "num_blocks": int(ns.compiled_num_blocks()),
            "block_size": int(ns.compiled_block_size()),
            "lm_num_blocks": int(ns.compiled_lm_num_blocks()),
            "lm_block_size": int(ns.compiled_lm_block_size()),
            "lm_int8": int(ns.compiled_lm_int8()),
            "lm_dp4a": int(ns.compiled_lm_dp4a()),
            "layer_int8": int(ns.compiled_layer_int8()),
            "layer_dp4a": int(ns.compiled_layer_dp4a()),
            "splitkv": int(ns.compiled_splitkv()),
            "splitkv_min_context": int(ns.compiled_splitkv_min_context()),
        }
        requested = {
            "num_blocks": int(os.environ.get("MEGAKERNEL_NUM_BLOCKS", "82")),
            "block_size": int(os.environ.get("MEGAKERNEL_BLOCK_SIZE", "1024")),
            "lm_num_blocks": int(os.environ.get("MEGAKERNEL_LM_NUM_BLOCKS", "8192")),
            "lm_block_size": int(os.environ.get("MEGAKERNEL_LM_BLOCK_SIZE", "256")),
            "lm_int8": int(os.environ.get("MEGAKERNEL_LM_INT8", "0")),
            "lm_dp4a": int(os.environ.get("MEGAKERNEL_LM_DP4A", "0")),
            "layer_int8": int(os.environ.get("MEGAKERNEL_LAYER_INT8", "0")),
            "layer_dp4a": int(os.environ.get("MEGAKERNEL_LAYER_DP4A", "0")),
            "splitkv": int(os.environ.get("MEGAKERNEL_DECODE_SPLITKV", "0")),
            "splitkv_min_context": int(os.environ.get("MEGAKERNEL_DECODE_SPLITKV_MIN_CONTEXT", "512")),
        }
        mismatches = [
            f"{k}: runtime={requested[k]} compiled={_compiled_config[k]}"
            for k in requested if requested[k] != _compiled_config[k]
        ]
        if mismatches:
            raise RuntimeError(
                "megakernel build/runtime configuration mismatch; set the same "
                "MEGAKERNEL_* values for build and execution: " + ", ".join(mismatches)
            )


def max_safe_decode_blocks() -> int:
    """Return the resident-block ceiling for the current CUDA device."""
    _load_op()
    return int(_max_safe_decode_blocks())


def set_decode_blocks(blocks: int):
    """Override decode blocks, clamped by the CUDA resident-block ceiling."""
    if blocks < 0:
        raise ValueError("blocks must be non-negative")
    _load_op()
    _set_decode_blocks(int(blocks))


def load_weights(model_name="Qwen/Qwen3.5-0.8B", verbose=True):
    """Load Qwen3.5-0.8B weights (bf16 on SM>=80, fp16 on SM<80)."""
    if not verbose:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hdtype = _half_dtype()
    dtype_name = "bf16" if hdtype == torch.bfloat16 else "fp16"

    if verbose:
        print(f"Loading {model_name} ({dtype_name})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=hdtype, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    def _w(key):
        """Get weight tensor, ensuring it's in the target half dtype."""
        return state[key].to(hdtype).contiguous()

    layer_data = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        lt = LAYER_TYPE[i]

        if lt == 1:
            # Full Attention: 11 pointers
            layer_data.append({
                "type": 1,
                "ptrs": [
                    _w(p + "input_layernorm.weight"),
                    _w(p + "self_attn.q_proj.weight"),
                    _w(p + "self_attn.k_proj.weight"),
                    _w(p + "self_attn.v_proj.weight"),
                    _w(p + "self_attn.q_norm.weight"),
                    _w(p + "self_attn.k_norm.weight"),
                    _w(p + "self_attn.o_proj.weight"),
                    _w(p + "post_attention_layernorm.weight"),
                    _w(p + "mlp.gate_proj.weight"),
                    _w(p + "mlp.up_proj.weight"),
                    _w(p + "mlp.down_proj.weight"),
                ]
            })
        else:
            # DeltaNet: 14 pointers
            layer_data.append({
                "type": 0,
                "ptrs": [
                    _w(p + "input_layernorm.weight"),
                    _w(p + "linear_attn.in_proj_qkv.weight"),
                    _w(p + "linear_attn.in_proj_z.weight"),
                    _w(p + "linear_attn.in_proj_b.weight"),
                    _w(p + "linear_attn.in_proj_a.weight"),
                    _w(p + "linear_attn.conv1d.weight"),
                    _w(p + "linear_attn.A_log"),
                    _w(p + "linear_attn.dt_bias"),
                    _w(p + "linear_attn.norm.weight"),
                    _w(p + "linear_attn.out_proj.weight"),
                    _w(p + "post_attention_layernorm.weight"),
                    _w(p + "mlp.gate_proj.weight"),
                    _w(p + "mlp.up_proj.weight"),
                    _w(p + "mlp.down_proj.weight"),
                ]
            })

    embed_weight = _w("model.embed_tokens.weight")
    final_norm_weight = _w("model.norm.weight")
    lm_head = state.get("lm_head.weight")
    if lm_head is None:
        lm_head = embed_weight
    else:
        lm_head = lm_head.to(hdtype).contiguous()

    weights = {
        "embed_weight": embed_weight,
        "final_norm_weight": final_norm_weight,
        "lm_head_weight": lm_head,
        "layer_data": layer_data,
    }

    del model
    torch.cuda.empty_cache()

    if verbose:
        total = sum(sum(t.numel() for t in ld["ptrs"]) for ld in layer_data) + lm_head.numel()
        print(f"{dtype_name.upper()} weights: {total/1e6:.1f}M params ({total*2/1e6:.0f} MB)")

    return weights, tokenizer


def _pack_layer_weights(layer_data):
    """Pack layer weights into device blob matching LayerWeights struct."""
    ptr_size = 8
    max_ptrs = 14
    header_size = 16
    struct_size = header_size + max_ptrs * ptr_size  # 128

    buf = bytearray(NUM_LAYERS * struct_size)
    for i in range(NUM_LAYERS):
        ld = layer_data[i]
        offset = i * struct_size
        struct.pack_into("iiii", buf, offset, ld["type"], 0, 0, 0)
        for j, tensor in enumerate(ld["ptrs"]):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, tensor.data_ptr())
        for j in range(len(ld["ptrs"]), max_ptrs):
            struct.pack_into("Q", buf, offset + header_size + j * ptr_size, 0)

    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


def _pack_decode_matrix(tensor):
    """Pack one [rows, cols] BF16 matrix as aligned per-row int8 + f32 scale."""
    rows, cols = tensor.shape
    f = tensor.float()
    scale = (f.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
    q = torch.round(f / scale[:, None]).clamp(-127, 127).to(torch.int8)
    packed = torch.empty((rows, cols + 16), dtype=torch.uint8, device="cuda")
    packed[:, :cols].copy_(q.view(torch.uint8))
    packed[:, cols:cols + 4].copy_(scale.view(torch.uint8).reshape(rows, 4))
    del f, scale, q
    return packed


class Decoder:
    """Stateful decoder for Qwen3.5-0.8B bf16 megakernel."""

    def __init__(self, weights=None, tokenizer=None,
                 model_name="Qwen/Qwen3.5-0.8B", verbose=True,
                 max_seq_len=MAX_SEQ_LEN, repetition_penalty=1.0,
                 decode_blocks=None):
        _load_op()
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be >= 1.0")
        if decode_blocks is not None and decode_blocks < 0:
            raise ValueError("decode_blocks must be non-negative")

        if weights is None:
            weights, tokenizer = load_weights(model_name, verbose=verbose)
        self.tokenizer = tokenizer
        self._position = 0
        self.max_seq_len = int(max_seq_len)
        self.repetition_penalty = float(repetition_penalty)
        self._weights = weights
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._lm_head_decode = self._lm_head_weight
        if os.environ.get("MEGAKERNEL_LM_INT8", "0") == "1":
            # Per-row symmetric int8 keeps the logits accurate while reducing
            # the decode-only vocabulary projection from 2 bytes/weight to
            # 1 byte/weight.  Keep the BF16 tensor for prefill.
            lm = self._lm_head_weight.float()
            scale = (lm.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
            q = torch.round(lm / scale[:, None]).clamp(-127, 127).to(torch.int8)
            # Four-byte scale plus padding to a 16-byte row stride, so every
            # vectorized load remains aligned for all vocabulary rows.
            packed = torch.empty((VOCAB_SIZE, HIDDEN_SIZE + 16), dtype=torch.uint8, device="cuda")
            packed[:, :HIDDEN_SIZE].copy_(q.view(torch.uint8))
            packed[:, HIDDEN_SIZE:HIDDEN_SIZE + 4].copy_(scale.view(torch.uint8).reshape(VOCAB_SIZE, 4))
            self._lm_head_decode = packed
            del lm, scale, q
        layer_data = weights["layer_data"]
        decode_layer_data = layer_data
        self._decode_packed_layers = []
        layer_int8 = os.environ.get("MEGAKERNEL_LAYER_INT8", "0") == "1"
        if layer_int8:
            # Only projection matrices are packed. Norms, convolution filters,
            # and recurrent constants remain BF16/f32 and retain their exact
            # state-update behavior.
            matrix_cols = {
                0: [1, 2, 3, 4, 9, 11, 12, 13],
                1: [1, 2, 3, 6, 8, 9, 10],
            }
            decode_layer_data = []
            for ld in layer_data:
                ptrs = list(ld["ptrs"])
                for j in matrix_cols[ld["type"]]:
                    packed = _pack_decode_matrix(ptrs[j])
                    self._decode_packed_layers.append(packed)
                    ptrs[j] = packed
                decode_layer_data.append({"type": ld["type"], "ptrs": ptrs})
        # Prefill always consumes BF16 matrices; decode may use the optional
        # packed int8 projection table.
        self._layer_weights_packed_bf16 = _pack_layer_weights(layer_data)
        self._layer_weights_packed = _pack_layer_weights(decode_layer_data)
        _set_decode_blocks(0 if decode_blocks is None else int(decode_blocks))

        bf16 = dict(dtype=_half_dtype(), device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")

        n_fa = sum(1 for t in LAYER_TYPE if t == 1)
        self._fa_k_cache = torch.zeros(n_fa, FA_NUM_KV_HEADS, self.max_seq_len, FA_HEAD_DIM, **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)

        n_dn = sum(1 for t in LAYER_TYPE if t == 0)
        self._dn_states = torch.zeros(n_dn, DN_NUM_HEADS, DN_KEY_DIM, DN_VALUE_DIM, **f32)
        self._conv_bufs = torch.zeros(n_dn, DN_CONV_CHANNELS, DN_CONV_KERNEL, **f32)

        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        max_scratch = max(FA_QPROJ_SIZE, DN_CONV_CHANNELS, HIDDEN_SIZE * 8 + INTERMEDIATE_SIZE)
        # Reserve a reusable tail for split-KV's one-pass per-block partials.
        # _load_op() has already checked that this runtime NUM_BLOCKS matches
        # the extension's compile-time stride.
        activation_stride = HIDDEN_SIZE + int(os.environ.get("MEGAKERNEL_NUM_BLOCKS", "82")) * FA_NUM_Q_HEADS * (FA_HEAD_DIM + 2)
        self._activations = torch.empty(max(max_scratch, activation_stride), **f32)
        self._residual = torch.empty(HIDDEN_SIZE, **bf16)
        self._qkv_scratch = torch.empty(max(FA_QPROJ_SIZE, DN_CONV_CHANNELS), **f32)
        self._kv_scratch = torch.empty(FA_KV_SIZE * 2, **f32)
        self._attn_out = torch.empty(max(FA_Q_SIZE, DN_V_SIZE), **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._z_scratch = torch.empty(DN_V_SIZE, **f32)
        self._beta_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._alpha_scratch = torch.empty(DN_NUM_HEADS, **f32)
        self._normalized = torch.empty(HIDDEN_SIZE, **f32)

        self._barrier_counter = torch.zeros(1, **u32)
        self._barrier_generation = torch.zeros(1, **u32)
        # The decode LM head can use more than the historical 512 blocks;
        # retain enough reduction storage for the selected build.
        # The decode extension's default LM grid is 8192 blocks.  Keep the
        # reduction buffers at least as large as the compiled grid; a smaller
        # historical default silently let the vocabulary kernel write past the
        # end of these tensors at long context.
        lm_blocks = max(1024, int(os.environ.get("MEGAKERNEL_LM_NUM_BLOCKS", "8192")))
        self._block_max_vals = torch.empty(lm_blocks, **f32)
        self._block_max_idxs = torch.empty(lm_blocks, **i32)
        self._lm_sync_counter = torch.zeros(1, **u32)
        self._seen_token_mask = torch.zeros(VOCAB_SIZE, **f32)
        self._out_token = torch.empty(1, **i32)
        # Device-side token handoff for decode_many; avoids a Python .item()
        # synchronization between consecutive greedy decode kernels.
        self._decode_token_buffer = torch.empty(1, **i32)

        # Pre-pack fused weights for the chunk-parallel prefill kernel:
        # one cuBLAS GEMM per layer instead of three (FA QKV) / two (MLP gate+up).
        layer_data = weights["layer_data"]
        fa_qkv_list = []
        for li in range(NUM_LAYERS):
            ld = layer_data[li]
            if ld['type'] == 1:
                q = ld['ptrs'][1]; k = ld['ptrs'][2]; v = ld['ptrs'][3]
                fa_qkv_list.append(torch.cat([q, k, v], dim=0))
        self._fused_fa_qkv = torch.stack(fa_qkv_list, dim=0).contiguous()
        gate_up_list = []
        for li in range(NUM_LAYERS):
            ld = layer_data[li]
            if ld['type'] == 0:
                g = ld['ptrs'][11]; u = ld['ptrs'][12]
            else:
                g = ld['ptrs'][8]; u = ld['ptrs'][9]
            gate_up_list.append(torch.cat([g, u], dim=0))
        self._fused_gate_up = torch.stack(gate_up_list, dim=0).contiguous()
        self._prefill_buffers = None
        self._prefill_buffer_tokens = 0
        self._prefill_graph_cache = {}

    def alloc_prefill_scratch(self, S: int):
        """Allocate per-prefill scratch buffers for the chunk-parallel kernel.
        Buffers depend on S (sequence length); call once per distinct S."""
        f32 = dict(dtype=torch.float32, device="cuda")
        S_pad = ((S + 31) // 32) * 32
        # Attention reuses dn_pre_qkv for an S×S score matrix and a trailing
        # S×head-dim workspace. Keep DeltaNet's projection footprint as the
        # lower bound for short prompts.
        prefill_aux_width = max(DN_CONV_CHANNELS, S + FA_HEAD_DIM)
        batched_attn = os.environ.get("MEGAKERNEL_PREFILL_ATTN_BATCH", "0").strip().lower() not in ("0", "false", "no", "off")
        if batched_attn:
            try:
                attn_block = max(1, int(os.environ.get("MEGAKERNEL_PREFILL_ATTN_BLOCK", "2048")))
            except ValueError:
                attn_block = 2048
            attn_block = min(attn_block, S)
            # Base score matrix + reusable four-head score and output tails.
            base_elems = max(S * prefill_aux_width, S * S)
            prefill_elems = base_elems + 4 * attn_block * S + 4 * attn_block * FA_HEAD_DIM
        else:
            prefill_elems = S * prefill_aux_width
        return dict(
            dn_pre_qkv=torch.empty(prefill_elems, **f32),
            dn_u_scratch=torch.empty(S_pad * DN_NUM_HEADS * 128, **f32),
            dn_w_scratch=torch.empty(S_pad * DN_NUM_HEADS * 128, **f32),
            dn_cs_scratch=torch.empty(S_pad * DN_NUM_HEADS, **f32),
        )

    def _ensure_prefill_buffers(self, max_tokens: int):
        """Allocate reusable BF16 prefill buffers for up to max_tokens."""
        if self._prefill_buffers is not None and self._prefill_buffer_tokens >= max_tokens:
            return self._prefill_buffers
        bf16 = dict(dtype=_half_dtype(), device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        mx = max(DN_CONV_CHANNELS, FA_QPROJ_SIZE, INTERMEDIATE_SIZE)
        mlp_scratch_width = max(INTERMEDIATE_SIZE, max_tokens)
        if os.environ.get("MEGAKERNEL_PREFILL_ATTN_BATCH", "0").strip().lower() not in ("0", "false", "no", "off"):
            try:
                attn_block = max(1, int(os.environ.get("MEGAKERNEL_PREFILL_ATTN_BLOCK", "2048")))
            except ValueError:
                attn_block = 2048
            mlp_scratch_width = max(mlp_scratch_width, 4 * min(attn_block, max_tokens))
        buffers = dict(
            hidden=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            residual=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            normalized=torch.empty(max_tokens * HIDDEN_SIZE, **bf16),
            proj_buf=torch.empty(max_tokens * mx, **bf16),
            proj_buf2=torch.empty(max_tokens * mx, **bf16),
            attn_buf=torch.empty(max_tokens * max(FA_Q_SIZE, FA_KV_SIZE), **bf16),
            mlp_buf=torch.empty(max_tokens * mlp_scratch_width, **bf16),
            dn_out_buf=torch.empty(max_tokens * DN_V_SIZE, **bf16),
            beta_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
            alpha_buf=torch.empty(max_tokens * DN_NUM_HEADS, **f32),
            final_normed=torch.empty(HIDDEN_SIZE, **bf16),
            hidden_bf16_out=torch.empty(HIDDEN_SIZE, **bf16),
            lm_bmv=torch.empty(1024, **f32),
            lm_bmi=torch.empty(1024, **i32),
        )
        buffers.update(self.alloc_prefill_scratch(max_tokens))
        self._prefill_buffers = buffers
        self._prefill_buffer_tokens = max_tokens
        return buffers

    def _run_prefill_bf16(self, token_ids: torch.Tensor, buffers):
        _prefill(
            self._out_token, token_ids,
            self._embed_weight, self._layer_weights_packed_bf16,
            self._final_norm_weight, self._lm_head_weight,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            buffers["hidden"], buffers["residual"], buffers["normalized"],
            buffers["proj_buf"], buffers["proj_buf2"], buffers["attn_buf"],
            buffers["mlp_buf"], buffers["dn_out_buf"], buffers["beta_buf"],
            buffers["alpha_buf"], buffers["dn_pre_qkv"], buffers["dn_u_scratch"],
            buffers["dn_w_scratch"], buffers["dn_cs_scratch"],
            self._fused_fa_qkv, self._fused_gate_up,
            buffers["final_normed"], buffers["hidden_bf16_out"],
            buffers["lm_bmv"], buffers["lm_bmi"], self.max_seq_len,
        )
        self._hidden.copy_(buffers["hidden_bf16_out"])

    def _prefill_graph_enabled(self):
        value = os.environ.get("MEGAKERNEL_PREFILL_GRAPH", "0").strip().lower()
        return value not in ("0", "false", "no", "off")

    def _build_prefill_graph(self, prompt_len: int):
        buffers = self._ensure_prefill_buffers(prompt_len)
        static_ids = torch.empty(prompt_len, dtype=torch.int32, device="cuda")
        warmup_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            self.reset()
            static_ids.zero_()
            self._run_prefill_bf16(static_ids, buffers)
        warmup_stream.synchronize()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        with torch.cuda.graph(graph):
            self.reset()
            self._run_prefill_bf16(static_ids, buffers)
        state = {"graph": graph, "token_ids": static_ids, "buffers": buffers}
        self._prefill_graph_cache[prompt_len] = state
        return state

    def prefill_tokens(self, token_ids: torch.Tensor) -> int:
        """Run fused BF16 prompt prefill and return the first generated token."""
        if token_ids.device.type != "cuda" or token_ids.dtype != torch.int32 or token_ids.dim() != 1:
            raise TypeError("token_ids must be a 1D CUDA int32 tensor")
        prompt_len = int(token_ids.numel())
        if prompt_len <= 0 or prompt_len > self.max_seq_len:
            raise ValueError(f"prompt length must be in [1, {self.max_seq_len}]")
        if self._prefill_graph_enabled():
            state = self._prefill_graph_cache.get(prompt_len)
            if state is None:
                state = self._build_prefill_graph(prompt_len)
            state["token_ids"].copy_(token_ids)
            state["graph"].replay()
        else:
            self.reset()
            self._run_prefill_bf16(token_ids, self._ensure_prefill_buffers(prompt_len))
        self._position = prompt_len
        return self._out_token.item()

    def step(self, token_id: int) -> int:
        """Decode one token. Returns next token id."""
        if self._position >= self.max_seq_len:
            raise ValueError(f"position {self._position} exceeds max_seq_len={self.max_seq_len}")
        _decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_decode,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter,
            self._seen_token_mask, self.repetition_penalty,
            self._position, self.max_seq_len,
        )
        self._position += 1
        return self._out_token.item()

    def step_many(self, token_id: int, num_steps: int) -> torch.Tensor:
        """Decode several greedy steps without per-token host synchronization.

        The returned CUDA int32 tensor contains the generated token IDs. The
        model state advances by ``num_steps``; sampling remains intentionally
        outside this fused path.
        """
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if num_steps == 0:
            return torch.empty(0, dtype=torch.int32, device="cuda")
        if self._position + num_steps > self.max_seq_len:
            raise ValueError("decode span exceeds max_seq_len")
        output_tokens = torch.empty(num_steps, dtype=torch.int32, device="cuda")
        _decode_many(
            output_tokens, self._decode_token_buffer, int(token_id),
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_decode,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter, self._seen_token_mask,
            self.repetition_penalty, self._position, self.max_seq_len,
        )
        self._position += num_steps
        return output_tokens

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()
        self._seen_token_mask.zero_()

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        token_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
        next_id = self.prefill_tokens(token_ids)
        out = [next_id]
        eos = self.tokenizer.eos_token_id
        # Greedy generation does not need a host synchronization for every
        # token.  Chain bounded chunks on the CUDA stream, then inspect the
        # returned device token vector once per chunk.  A bounded default keeps
        # EOS latency low; set MEGAKERNEL_GENERATE_CHUNK=1 to recover the
        # token-at-a-time streaming behavior.
        remaining = max(0, max_tokens - 1)
        chunk_size = max(1, int(os.environ.get("MEGAKERNEL_GENERATE_CHUNK", "32")))
        while remaining:
            count = min(remaining, chunk_size)
            generated = self.step_many(next_id, count).cpu().tolist()
            for token in generated:
                if token == eos:
                    return self.tokenizer.decode(out, skip_special_tokens=True)
                out.append(token)
            next_id = generated[-1]
            remaining -= count
        return self.tokenizer.decode(out, skip_special_tokens=True)


class BatchDecoder:
    """Equal-length batch decode wrapper with independent recurrent/KV state.

    Prefill currently uses the validated single-sequence kernel once per item;
    decode runs all items through one batch-strided CUDA launch. Keeping the
    fallback explicit avoids silently mixing recurrent state between prompts.
    """

    def __init__(self, batch_size, weights=None, tokenizer=None,
                 model_name="Qwen/Qwen3.5-0.8B", verbose=True,
                 max_seq_len=MAX_SEQ_LEN, repetition_penalty=1.0,
                 decode_blocks=None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        self.base = Decoder(
            weights=weights, tokenizer=tokenizer, model_name=model_name,
            verbose=verbose, max_seq_len=max_seq_len,
            repetition_penalty=repetition_penalty,
            decode_blocks=decode_blocks,
        )
        self.tokenizer = self.base.tokenizer
        self.max_seq_len = self.base.max_seq_len
        self.repetition_penalty = self.base.repetition_penalty
        self._position = 0
        bf16 = dict(dtype=_half_dtype(), device="cuda")
        f32 = dict(dtype=torch.float32, device="cuda")
        i32 = dict(dtype=torch.int32, device="cuda")
        u32 = dict(dtype=torch.uint32, device="cuda")
        B = self.batch_size
        def batch_like(t):
            return torch.empty((B, *t.shape), dtype=t.dtype, device=t.device)

        self._fa_k_cache = torch.zeros((B, *self.base._fa_k_cache.shape), **bf16)
        self._fa_v_cache = torch.zeros_like(self._fa_k_cache)
        self._dn_states = torch.zeros((B, *self.base._dn_states.shape), **f32)
        self._conv_bufs = torch.zeros((B, *self.base._conv_bufs.shape), **f32)
        self._hidden = batch_like(self.base._hidden)
        self._activations = batch_like(self.base._activations)
        self._residual = batch_like(self.base._residual)
        self._qkv_scratch = batch_like(self.base._qkv_scratch)
        self._kv_scratch = batch_like(self.base._kv_scratch)
        self._attn_out = batch_like(self.base._attn_out)
        self._mlp_inter = batch_like(self.base._mlp_inter)
        self._z_scratch = batch_like(self.base._z_scratch)
        self._beta_scratch = batch_like(self.base._beta_scratch)
        self._alpha_scratch = batch_like(self.base._alpha_scratch)
        self._normalized = batch_like(self.base._normalized)
        self._barrier_counter = torch.zeros(B, **u32)
        self._barrier_generation = torch.zeros(B, **u32)
        self._block_max_vals = torch.empty((B, self.base._block_max_vals.numel()), **f32)
        self._block_max_idxs = torch.empty((B, self.base._block_max_idxs.numel()), **i32)
        self._lm_sync_counter = torch.zeros(B, **u32)
        self._seen_token_mask = torch.zeros((B, VOCAB_SIZE), **f32)
        self._out_tokens = torch.empty(B, **i32)

    def prefill_batch(self, token_ids):
        """Prefill equal-length prompts and return one first token per item."""
        if isinstance(token_ids, (list, tuple)):
            if len(token_ids) != self.batch_size:
                raise ValueError("token_ids length must equal batch_size")
            token_ids = torch.stack(tuple(token_ids))
        if (token_ids.device.type != "cuda" or token_ids.dtype != torch.int32 or
                token_ids.dim() != 2 or token_ids.shape[0] != self.batch_size):
            raise TypeError("token_ids must be a [batch_size, sequence] CUDA int32 tensor")
        S = int(token_ids.shape[1])
        if S <= 0 or S > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        first = []
        for b in range(self.batch_size):
            first.append(self.base.prefill_tokens(token_ids[b]))
            self._fa_k_cache[b].copy_(self.base._fa_k_cache)
            self._fa_v_cache[b].copy_(self.base._fa_v_cache)
            self._dn_states[b].copy_(self.base._dn_states)
            self._conv_bufs[b].copy_(self.base._conv_bufs)
            self._hidden[b].copy_(self.base._hidden)
        self._position = S
        return torch.tensor(first, dtype=torch.int32, device="cuda")

    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Decode one token for every sequence in the batch."""
        if (token_ids.device.type != "cuda" or token_ids.dtype != torch.int32 or
                token_ids.dim() != 1 or token_ids.numel() != self.batch_size):
            raise TypeError("token_ids must be a [batch_size] CUDA int32 tensor")
        if self._position >= self.max_seq_len:
            raise ValueError("position exceeds max_seq_len")
        _decode_batch(
            self._out_tokens, token_ids,
            self.base._embed_weight, self.base._layer_weights_packed,
            self.base._final_norm_weight, self.base._lm_head_decode,
            self._fa_k_cache, self._fa_v_cache,
            self._dn_states, self._conv_bufs,
            self._hidden, self._activations, self._residual,
            self._qkv_scratch, self._kv_scratch, self._attn_out,
            self._mlp_inter, self._z_scratch, self._beta_scratch,
            self._alpha_scratch, self._normalized,
            self._barrier_counter, self._barrier_generation,
            self._block_max_vals, self._block_max_idxs,
            self._lm_sync_counter, self._seen_token_mask,
            self.repetition_penalty, self._position, self.max_seq_len,
        )
        self._position += 1
        return self._out_tokens

    def reset(self):
        self._position = 0
        self._fa_k_cache.zero_()
        self._fa_v_cache.zero_()
        self._dn_states.zero_()
        self._conv_bufs.zero_()
        self._seen_token_mask.zero_()
