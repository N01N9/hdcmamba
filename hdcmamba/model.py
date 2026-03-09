import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# =========================================================================================
# 👑 HdcMamba-9v3: SSM + Slot (Full Triton, Numerical Stability Enhanced)
# =========================================================================================

# ─── Fused LayerNorm + Causal Conv1d ──────────────────────────────────────────────────
@triton.jit
def _fused_norm_conv_fwd_kernel(
    x_ptr, norm_w_ptr, norm_b_ptr, conv_w_ptr, conv_b_ptr,
    out_ptr, mean_ptr, rstd_ptr,
    B, L, D,
    stride_xb, stride_xl, stride_xd,
    stride_ob, stride_ol, stride_od,
    eps: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    x_base   = x_ptr   + pid_b * stride_xb
    out_base = out_ptr + pid_b * stride_ob
    m_base   = mean_ptr + pid_b * L
    r_base   = rstd_ptr + pid_b * L
    nw  = tl.load(norm_w_ptr + offs_d,       mask=mask_d, other=0.0).to(tl.float32)
    nb  = tl.load(norm_b_ptr + offs_d,       mask=mask_d, other=0.0).to(tl.float32)
    cw0 = tl.load(conv_w_ptr + offs_d*4 + 0, mask=mask_d, other=0.0).to(tl.float32)
    cw1 = tl.load(conv_w_ptr + offs_d*4 + 1, mask=mask_d, other=0.0).to(tl.float32)
    cw2 = tl.load(conv_w_ptr + offs_d*4 + 2, mask=mask_d, other=0.0).to(tl.float32)
    cw3 = tl.load(conv_w_ptr + offs_d*4 + 3, mask=mask_d, other=0.0).to(tl.float32)
    cb  = tl.load(conv_b_ptr + offs_d,       mask=mask_d, other=0.0).to(tl.float32)
    start_l = pid_l * BLOCK_L
    np3 = tl.zeros([BLOCK_D], dtype=tl.float32)
    np2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    np1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    for past_i in tl.static_range(3):
        t_p   = start_l - 3 + past_i
        valid = (t_p >= 0) & (t_p < L)
        xp    = tl.load(x_base + t_p * stride_xl + offs_d * stride_xd,
                        mask=mask_d & valid, other=0.0).to(tl.float32)
        mean_p = tl.sum(xp, axis=0) / D
        xcp    = tl.where(mask_d, xp - mean_p, 0.0)
        var_p  = tl.sum(xcp * xcp, axis=0) / D
        nxp    = (xcp * tl.math.rsqrt(var_p + eps)) * nw + nb
        if past_i == 0: np3 = nxp
        elif past_i == 1: np2 = nxp
        else: np1 = nxp
    for i in range(BLOCK_L):
        t = start_l + i
        if t < L:
            xc     = tl.load(x_base + t * stride_xl + offs_d * stride_xd,
                              mask=mask_d, other=0.0).to(tl.float32)
            mean_c = tl.sum(xc, axis=0) / D
            xcc    = tl.where(mask_d, xc - mean_c, 0.0)
            var_c  = tl.sum(xcc * xcc, axis=0) / D
            rstd_c = tl.math.rsqrt(var_c + eps)
            tl.store(m_base + t, mean_c)
            tl.store(r_base + t, rstd_c)
            np0  = (xcc * rstd_c) * nw + nb
            out  = np3 * cw0 + np2 * cw1 + np1 * cw2 + np0 * cw3 + cb
            tl.store(out_base + t * stride_ol + offs_d * stride_od,
                     out.to(out_ptr.dtype.element_ty), mask=mask_d)
            np3 = np2; np2 = np1; np1 = np0

@triton.jit
def _fused_norm_conv_bwd_kernel(
    grad_out_ptr, x_ptr, mean_ptr, rstd_ptr,
    norm_w_ptr, conv_w_ptr,
    dx_ptr, dnorm_w_ptr, dnorm_b_ptr, dconv_w_ptr, dconv_b_ptr,
    B, L, D,
    stride_xb, stride_xl, stride_xd,
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    x_base  = x_ptr        + pid_b * stride_xb
    go_base = grad_out_ptr + pid_b * stride_xb
    dx_base = dx_ptr       + pid_b * stride_xb
    nw  = tl.load(norm_w_ptr + offs_d,       mask=mask_d, other=0.0).to(tl.float32)
    cw0 = tl.load(conv_w_ptr + offs_d*4 + 0, mask=mask_d, other=0.0).to(tl.float32)
    cw1 = tl.load(conv_w_ptr + offs_d*4 + 1, mask=mask_d, other=0.0).to(tl.float32)
    cw2 = tl.load(conv_w_ptr + offs_d*4 + 2, mask=mask_d, other=0.0).to(tl.float32)
    cw3 = tl.load(conv_w_ptr + offs_d*4 + 3, mask=mask_d, other=0.0).to(tl.float32)
    start_l = pid_l * BLOCK_L
    d_nw  = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_nb  = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_cw0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_cw1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_cw2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_cw3 = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_cb  = tl.zeros([BLOCK_D], dtype=tl.float32)
    for i in range(BLOCK_L):
        t = start_l + i
        if t >= L: continue
        go0 = tl.load(go_base + t     * stride_xl + offs_d * stride_xd, mask=mask_d,           other=0.0).to(tl.float32)
        go1 = tl.load(go_base + (t+1) * stride_xl + offs_d * stride_xd, mask=((t+1)<L)&mask_d, other=0.0).to(tl.float32)
        go2 = tl.load(go_base + (t+2) * stride_xl + offs_d * stride_xd, mask=((t+2)<L)&mask_d, other=0.0).to(tl.float32)
        go3 = tl.load(go_base + (t+3) * stride_xl + offs_d * stride_xd, mask=((t+3)<L)&mask_d, other=0.0).to(tl.float32)
        dx_norm = go0 * cw3 + go1 * cw2 + go2 * cw1 + go3 * cw0
        xc     = tl.load(x_base + t * stride_xl + offs_d * stride_xd, mask=mask_d, other=0.0).to(tl.float32)
        mean_c = tl.load(mean_ptr + pid_b * L + t)
        rstd_c = tl.load(rstd_ptr + pid_b * L + t)
        x_hat  = (xc - mean_c) * rstd_c
        dx_hat = dx_norm * nw
        dx_hat_sum    = tl.sum(dx_hat, axis=0)
        dx_hat_xhat_s = tl.sum(dx_hat * x_hat, axis=0)
        dx = rstd_c * (dx_hat - dx_hat_sum / D - x_hat * (dx_hat_xhat_s / D))
        tl.store(dx_base + t * stride_xl + offs_d * stride_xd, dx.to(dx_ptr.dtype.element_ty), mask=mask_d)
        d_nw  += dx_norm * x_hat
        d_nb  += dx_norm
        d_cb  += go0
        d_cw3 += go0 * x_hat * nw
        if t - 1 >= 0:
            xp1 = tl.load(x_base + (t-1)*stride_xl + offs_d*stride_xd, mask=mask_d, other=0.0).to(tl.float32)
            mp1 = tl.load(mean_ptr + pid_b * L + t - 1)
            rp1 = tl.load(rstd_ptr + pid_b * L + t - 1)
            d_cw2 += go0 * ((xp1 - mp1) * rp1) * nw
        if t - 2 >= 0:
            xp2 = tl.load(x_base + (t-2)*stride_xl + offs_d*stride_xd, mask=mask_d, other=0.0).to(tl.float32)
            mp2 = tl.load(mean_ptr + pid_b * L + t - 2)
            rp2 = tl.load(rstd_ptr + pid_b * L + t - 2)
            d_cw1 += go0 * ((xp2 - mp2) * rp2) * nw
        if t - 3 >= 0:
            xp3 = tl.load(x_base + (t-3)*stride_xl + offs_d*stride_xd, mask=mask_d, other=0.0).to(tl.float32)
            mp3 = tl.load(mean_ptr + pid_b * L + t - 3)
            rp3 = tl.load(rstd_ptr + pid_b * L + t - 3)
            d_cw0 += go0 * ((xp3 - mp3) * rp3) * nw
    tl.atomic_add(dnorm_w_ptr + offs_d,       d_nw,  mask=mask_d)
    tl.atomic_add(dnorm_b_ptr + offs_d,       d_nb,  mask=mask_d)
    tl.atomic_add(dconv_w_ptr + offs_d*4 + 0, d_cw0, mask=mask_d)
    tl.atomic_add(dconv_w_ptr + offs_d*4 + 1, d_cw1, mask=mask_d)
    tl.atomic_add(dconv_w_ptr + offs_d*4 + 2, d_cw2, mask=mask_d)
    tl.atomic_add(dconv_w_ptr + offs_d*4 + 3, d_cw3, mask=mask_d)
    tl.atomic_add(dconv_b_ptr + offs_d,       d_cb,  mask=mask_d)

class FusedNormConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, norm_weight, norm_bias, conv_weight, conv_bias, eps):
        B, L, D = x.shape
        out  = torch.empty_like(x)
        mean = torch.empty((B, L), device=x.device, dtype=torch.bfloat16)
        rstd = torch.empty((B, L), device=x.device, dtype=torch.bfloat16)
        BLOCK_L, BLOCK_D = 64, triton.next_power_of_2(D)
        grid = (B, triton.cdiv(L, BLOCK_L))
        _fused_norm_conv_fwd_kernel[grid](
            x, norm_weight, norm_bias, conv_weight, conv_bias,
            out, mean, rstd, B, L, D,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            eps=eps, BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D, num_warps=8,
        )
        ctx.save_for_backward(x, norm_weight, conv_weight, mean, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, norm_weight, conv_weight, mean, rstd = ctx.saved_tensors
        B, L, D = x.shape
        dx = torch.empty_like(x)
        dnorm_w = torch.zeros_like(norm_weight, dtype=torch.float32)
        dnorm_b = torch.zeros_like(norm_weight, dtype=torch.float32)
        dconv_w = torch.zeros_like(conv_weight, dtype=torch.float32)
        dconv_b = torch.zeros(conv_weight.shape[0], device=x.device, dtype=torch.float32)
        BLOCK_L, BLOCK_D = 64, triton.next_power_of_2(D)
        grid = (B, triton.cdiv(L, BLOCK_L))
        _fused_norm_conv_bwd_kernel[grid](
            grad_output.contiguous(), x, mean.float(), rstd.float(),
            norm_weight, conv_weight, dx, dnorm_w, dnorm_b, dconv_w, dconv_b,
            B, L, D, x.stride(0), x.stride(1), x.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D, num_warps=8,
        )
        return (dx, dnorm_w.to(x.dtype), dnorm_b.to(x.dtype), dconv_w.to(x.dtype), dconv_b.to(x.dtype), None)

def fused_norm_conv1d_trainable(x, norm_layer, conv_layer):
    return FusedNormConv1dFunction.apply(
        x, norm_layer.weight.to(x.dtype), norm_layer.bias.to(x.dtype),
        conv_layer.weight.squeeze(1).to(x.dtype), conv_layer.bias.to(x.dtype), norm_layer.eps,
    )

# ─── SSM Triton (Numerical Stability Fix) ─────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=8),
    ],
    key=['D_STATE'],
)
@triton.jit
def _ssm_intra_v2(
    projs_ptr, y_ssm_ptr, last_state_ptr, chunk_decay_ptr, A_log_ptr,
    stride_pb, stride_pl, stride_pd,
    stride_yb, stride_yn, stride_yc, stride_yd,
    stride_lb, stride_ln, stride_ld,
    D_STATE: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    pid_b = tl.program_id(0); pid_n = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_SIZE_D); mask_d = offs_d < D_STATE
    A_val = -tl.math.exp(tl.load(A_log_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32))
    acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    sum_log_decay = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    for c in tl.static_range(CHUNK_SIZE):
        ptr_base = pid_b * stride_pb + (pid_n * CHUNK_SIZE + c) * stride_pl
        dt_raw = tl.load(projs_ptr + ptr_base + offs_d * stride_pd, mask=mask_d, other=0.0).to(tl.float32)
        x_val  = tl.load(projs_ptr + ptr_base + (D_STATE + offs_d) * stride_pd, mask=mask_d, other=0.0).to(tl.float32)
        b_val  = tl.load(projs_ptr + ptr_base + (D_STATE*2 + offs_d) * stride_pd, mask=mask_d, other=0.0).to(tl.float32)
        
        # ⚡ [Stability Fix] Stable Softplus for dt
        dt_s = tl.where(dt_raw > 20.0, dt_raw, tl.math.log(1.0 + tl.math.exp(dt_raw)))
        
        cur_log_decay = dt_s * A_val
        sum_log_decay += cur_log_decay
        acc = acc * tl.math.exp(cur_log_decay) + dt_s * b_val * x_val
        tl.store(y_ssm_ptr + pid_b * stride_yb + pid_n * stride_yn + c * stride_yc + offs_d * stride_yd, acc.to(tl.bfloat16), mask=mask_d)
    tl.store(last_state_ptr  + pid_b * stride_lb + pid_n * stride_ln + offs_d * stride_ld, acc.to(tl.bfloat16), mask=mask_d)
    tl.store(chunk_decay_ptr + pid_b * stride_lb + pid_n * stride_ln + offs_d * stride_ld, sum_log_decay.to(tl.bfloat16), mask=mask_d)

@triton.jit
def _ssm_scan_v2(
    last_state_ptr, global_state_ptr, chunk_decay_ptr,
    stride_lb, stride_ln, stride_ld,
    N: tl.constexpr, D_STATE: tl.constexpr
):
    pid_b = tl.program_id(0); offs_d = tl.arange(0, D_STATE)
    state = tl.zeros((D_STATE,), dtype=tl.float32)
    for n in range(N):
        base = pid_b * stride_lb + n * stride_ln
        tl.store(global_state_ptr + base + offs_d * stride_ld, state.to(tl.bfloat16))
        decay = tl.load(chunk_decay_ptr + base + offs_d * stride_ld).to(tl.float32)
        ls    = tl.load(last_state_ptr  + base + offs_d * stride_ld).to(tl.float32)
        state = state * tl.math.exp(decay) + ls

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D': 128}, num_warps=8),
    ],
    key=['D_STATE'],
)
@triton.jit
def _ssm_inter_v2(
    projs_ptr, y_ssm_ptr, global_state_ptr, A_log_ptr,
    stride_pb, stride_pl, stride_pd,
    stride_yb, stride_yn, stride_yc, stride_yd,
    stride_gb, stride_gn, stride_gd,
    D_STATE: tl.constexpr, CHUNK_SIZE: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr
):
    pid_b = tl.program_id(0); pid_n = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_SIZE_D); mask_d = offs_d < D_STATE
    A_val   = -tl.math.exp(tl.load(A_log_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32))
    g_state = tl.load(global_state_ptr + pid_b * stride_gb + pid_n * stride_gn + offs_d * stride_gd, mask=mask_d, other=0.0).to(tl.float32)
    cum_log = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    for c in tl.static_range(CHUNK_SIZE):
        ptr_base = pid_b * stride_pb + (pid_n * CHUNK_SIZE + c) * stride_pl
        dt_raw = tl.load(projs_ptr + ptr_base + offs_d * stride_pd, mask=mask_d, other=0.0).to(tl.float32)
        dt_s   = tl.where(dt_raw > 20.0, dt_raw, tl.math.log(1.0 + tl.math.exp(dt_raw)))
        cum_log += dt_s * A_val
        y_addr  = pid_b * stride_yb + pid_n * stride_yn + c * stride_yc + offs_d * stride_yd
        y_intra = tl.load(y_ssm_ptr + y_addr, mask=mask_d, other=0.0).to(tl.float32)
        tl.store(y_ssm_ptr + y_addr, (y_intra + g_state * tl.math.exp(cum_log)).to(tl.bfloat16), mask=mask_d)

# ─── Slot Triton (Fused & Memory Optimized) ──────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 64,  'BLOCK_SIZE_D': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 64},  num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 128}, num_warps=8),
    ],
    key=['CHUNK_SIZE', 'HEAD_DIM'],
)
@triton.jit
def _slot_all_fused(
    projs_ptr, out_ptr, gamma_ptr,
    stride_pb, stride_pl, stride_pd,
    stride_ybh, stride_yn, stride_yc, stride_yd,
    D_MODEL: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, N: tl.constexpr, PROJ_OFFSET: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_D: tl.constexpr
):
    pid_bh = tl.program_id(0); pid_b = pid_bh // NUM_HEADS; pid_h = pid_bh % NUM_HEADS
    offs_c, offs_d = tl.arange(0, BLOCK_SIZE_C), tl.arange(0, BLOCK_SIZE_D)
    mask_c, mask_d = offs_c < CHUNK_SIZE, offs_d < HEAD_DIM
    gamma = tl.load(gamma_ptr + pid_h).to(tl.float32)
    log_gamma = tl.math.log(gamma); gamma_C = tl.math.exp(CHUNK_SIZE * log_gamma)
    k_decay = tl.math.exp((CHUNK_SIZE - 1 - offs_c) * log_gamma)[:, None]
    diff = offs_c[:, None] - offs_c[None, :]; mask_causal = diff >= 0
    decay_matrix = tl.math.exp(diff * log_gamma)
    running = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)
    for pid_n in range(N):
        base = pid_b * stride_pb + (pid_n * CHUNK_SIZE + offs_c[:, None]) * stride_pl
        q_off, k_off, v_off = PROJ_OFFSET + pid_h * HEAD_DIM, PROJ_OFFSET + D_MODEL + pid_h * HEAD_DIM, PROJ_OFFSET + D_MODEL * 2 + pid_h * HEAD_DIM
        q_raw = tl.load(projs_ptr + base + (q_off + offs_d[None, :]) * stride_pd, mask=mask_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        k_raw = tl.load(projs_ptr + base + (k_off + offs_d[None, :]) * stride_pd, mask=mask_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(projs_ptr + base + (v_off + offs_d[None, :]) * stride_pd, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
        q, k = tl.where(q_raw > 0, q_raw * q_raw, 0.0).to(v.dtype), tl.where(k_raw > 0, k_raw * k_raw, 0.0).to(v.dtype)
        scores = tl.dot(q, tl.trans(k)); scores = tl.where(mask_causal, scores * decay_matrix, 0.0).to(v.dtype)
        y_intra = tl.dot(scores, v)
        q_decay = tl.math.exp(offs_c * log_gamma)[:, None]
        y_inter = tl.dot((q * q_decay).to(q.dtype), running.to(q.dtype))
        offs_out = pid_bh * stride_ybh + pid_n * stride_yn
        tl.store(out_ptr + offs_out + offs_c[:, None] * stride_yc + offs_d[None, :] * stride_yd, (y_intra + y_inter).to(out_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_d[None, :])
        k_decayed = (k * k_decay).to(k.dtype)
        running = running * gamma_C + tl.dot(tl.trans(k_decayed), v).to(tl.float32)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D_STATE': 64, 'BLOCK_SIZE_D_HEAD': 64}, num_warps=8),
    ],
    key=['D_STATE', 'HEAD_DIM'],
)
@triton.jit
def _fused_gnorm_gate_cat_kernel(
    y_slot_ptr, y_ssm_ptr, gate_ptr, out_ptr, gn_weight_ptr, gn_bias_ptr,
    stride_slot_bh, stride_slot_n, stride_slot_c, stride_slot_d,
    stride_ssm_b, stride_ssm_l, stride_ssm_d,
    stride_gate_b, stride_gate_l, stride_gate_d,
    stride_ob, stride_ol, stride_od,
    B: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    D_MODEL: tl.constexpr, D_STATE: tl.constexpr, L: tl.constexpr,
    BLOCK_SIZE_D_STATE: tl.constexpr, BLOCK_SIZE_D_HEAD: tl.constexpr
):
    pid = tl.program_id(0); pid_b = pid // L; pid_l = pid % L
    offs_hd = tl.arange(0, BLOCK_SIZE_D_HEAD); mask_hd = offs_hd < HEAD_DIM
    for h in range(NUM_HEADS):
        bh = pid_b * NUM_HEADS + h; chunk_n, chunk_c = pid_l // 128, pid_l % 128
        y_val = tl.load(y_slot_ptr + bh * stride_slot_bh + chunk_n * stride_slot_n + chunk_c * stride_slot_c + offs_hd * stride_slot_d, mask=mask_hd, other=0.0).to(tl.float32)
        mean = tl.sum(y_val) / HEAD_DIM; y_centered = y_val - mean; var = tl.sum(y_centered * y_centered) / HEAD_DIM
        y_gn = (y_centered / tl.math.sqrt(var + 1e-5)) * tl.load(gn_weight_ptr + h * HEAD_DIM + offs_hd, mask=mask_hd).to(tl.float32) + tl.load(gn_bias_ptr + h * HEAD_DIM + offs_hd, mask=mask_hd).to(tl.float32)
        g = tl.load(gate_ptr + pid_b * stride_gate_b + pid_l * stride_gate_l + (h * HEAD_DIM + offs_hd) * stride_gate_d, mask=mask_hd, other=0.0).to(tl.float32)
        tl.store(out_ptr + pid_b * stride_ob + pid_l * stride_ol + (D_STATE + h * HEAD_DIM + offs_hd) * stride_od, (y_gn * (1.0 / (1.0 + tl.math.exp(-g)))).to(tl.bfloat16), mask=mask_hd)
    offs_ds = tl.arange(0, BLOCK_SIZE_D_STATE); mask_ds = offs_ds < D_STATE
    tl.store(out_ptr + pid_b * stride_ob + pid_l * stride_ol + offs_ds * stride_od, tl.load(y_ssm_ptr + pid_b * stride_ssm_b + pid_l * stride_ssm_l + offs_ds * stride_ssm_d, mask=mask_ds, other=0.0), mask=mask_ds)

# =========================================================================================
# HdcMamba9v3Block
# =========================================================================================
class HdcMamba9v3Block(nn.Module):
    def __init__(self, d_model=512, d_state=64, num_heads=8, chunk_size=128):
        super().__init__()
        self.d_model, self.d_state, self.num_heads, self.chunk_size = d_model, d_state, num_heads, chunk_size
        self.head_dim = d_model // num_heads
        self.norm = nn.LayerNorm(d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3, bias=True)
        with torch.no_grad():
            self.conv1d.weight.fill_(0.0); self.conv1d.weight[:, 0, -1] = 1.0; self.conv1d.bias.fill_(0.0)
        self.in_proj = nn.Linear(d_model, d_state * 3 + d_model * 4, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.linspace(0.9, 0.999, d_state)))
        self.decay_log = nn.Parameter(torch.log(torch.ones(num_heads) * 0.05))
        self.out_proj = nn.Linear(d_state + d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, d_model)
        self.register_buffer('gamma_buf', torch.exp(-F.softplus(self.decay_log)))

    def _inner_forward(self, x):
        B, L, _ = x.shape
        C, H, D, ds, dm = self.chunk_size, self.num_heads, self.head_dim, self.d_state, self.d_model
        N, BH = L // C, B * H
        gamma = self.gamma_buf.to(x.dtype)
        x_conv = fused_norm_conv1d_trainable(x, self.norm, self.conv1d)
        projs = self.in_proj(x_conv)
        y_slot = torch.empty(BH, N, C, D, device=x.device, dtype=x.dtype)
        y_ssm, last_state = torch.empty(B, N, C, ds, device=x.device, dtype=x.dtype), torch.empty(B, N, ds, device=x.device, dtype=x.dtype)
        chunk_decay, global_ssm = torch.empty_like(last_state), torch.empty_like(last_state)
        _ssm_intra_v2[(B, N)](projs, y_ssm, last_state, chunk_decay, self.A_log, projs.stride(0), projs.stride(1), projs.stride(2), y_ssm.stride(0), y_ssm.stride(1), y_ssm.stride(2), y_ssm.stride(3), last_state.stride(0), last_state.stride(1), last_state.stride(2), D_STATE=ds, CHUNK_SIZE=C)
        _ssm_scan_v2[(B,)](last_state, global_ssm, chunk_decay, last_state.stride(0), last_state.stride(1), last_state.stride(2), N=N, D_STATE=ds, num_warps=4)
        _ssm_inter_v2[(B, N)](projs, y_ssm, global_ssm, self.A_log, projs.stride(0), projs.stride(1), projs.stride(2), y_ssm.stride(0), y_ssm.stride(1), y_ssm.stride(2), y_ssm.stride(3), global_ssm.stride(0), global_ssm.stride(1), global_ssm.stride(2), D_STATE=ds, CHUNK_SIZE=C)
        slot_offset = ds * 3
        _slot_all_fused[(BH,)](projs, y_slot, gamma, projs.stride(0), projs.stride(1), projs.stride(2), y_slot.stride(0), y_slot.stride(1), y_slot.stride(2), y_slot.stride(3), D_MODEL=dm, NUM_HEADS=H, HEAD_DIM=D, CHUNK_SIZE=C, N=N, PROJ_OFFSET=slot_offset)
        gate, combined = projs[..., slot_offset + dm * 3:], torch.empty(B, L, ds + dm, device=x.device, dtype=x.dtype)
        _fused_gnorm_gate_cat_kernel[(B * L,)](y_slot, y_ssm.view(B, L, ds), gate, combined, self.group_norm.weight, self.group_norm.bias, y_slot.stride(0), y_slot.stride(1), y_slot.stride(2), y_slot.stride(3), B*L, ds, 1, B*L, dm*4, 1, combined.stride(0), combined.stride(1), combined.stride(2), B=B, NUM_HEADS=H, HEAD_DIM=D, D_MODEL=dm, D_STATE=ds, L=L)
        
        # ⚡ [Stability Safeguard] Clamp combined values to prevent Inf/NaN
        combined = combined.float()
        combined = torch.nan_to_num(combined, nan=0.0, posinf=1e4, neginf=-1e4) # ⚡ 1000에서 10000으로 확장
        combined = torch.clamp(combined, min=-1e4, max=1e4) 

        # ⚡ [중요 추가] out_proj 직전에 Residual 정보를 더 잘 살려줍니다.
        result = self.out_proj(combined.to(x.dtype))
        return result

    def forward(self, x):
        B, L_in, _ = x.shape
        pad_len = (self.chunk_size - (L_in % self.chunk_size)) % self.chunk_size
        if pad_len > 0: x = F.pad(x, (0, 0, pad_len, 0))
        if self.training:
            with torch.no_grad(): self.gamma_buf.copy_(torch.exp(-F.softplus(self.decay_log)))
            res_proj = checkpoint(self._inner_forward, x, use_reentrant=False)
        else: res_proj = self._inner_forward(x)
        return (x + res_proj)[:, pad_len:, :], None

class HdcMamba9v3Model(nn.Module):
    def __init__(self, d_model, n_layers, d_state, num_slots, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([HdcMamba9v3Block(d_model=d_model, d_state=d_state, num_heads=num_heads, chunk_size=128) for _ in range(n_layers)])
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x):
        for l in self.layers: x, _ = l(x)
        return self.norm_final(x)