import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# =========================================================================================
# 👑 HdcMamba-9v3: SSM + Slot (Full Triton, No Compile, Ultra-Minimal Launches)
#
# [메모리 선형증가 근본 원인 및 수정]
#
# checkpoint()가 해제하는 것: autograd activation (역전파용 중간값)
# checkpoint()가 해제 못 하는 것: Triton 커널의 I/O 버퍼 (연산 자체에 필요한 텐서)
#
# 진짜 범인 — "compute buffer" 5종:
#
#   ❌ projs       (B, L, 3264)       : in_proj 출력, 6개 커널이 모두 읽음
#   ❌ y_ssm       (B, L, d_state)    : intra→inter 전달용
#   ❌ y_slot      (BH, L, head_dim)  : pass1→pass3 전달용
#   ❌ chunk_state (BH, N, D, D)      : ← 핵심! N=L/C 에 선형
#   ❌ global_slot (BH, N, D, D)      : ← 핵심! N=L/C 에 선형
#
# 수정 전략:
#   chunk_state, global_slot: N 차원 제거 → _slot_pass1과 _slot_pass2를 fuse
#                              (BH, N, D, D) → running state (BH, D, D) 1개
#   projs, y_ssm, y_slot: 구조상 전체 L 필요 → 현재는 그대로 유지
#                          (추가 최적화: chunk-by-chunk 재설계 필요, 별도 작업)
# =========================================================================================

# ─── Fused LayerNorm + Causal Conv1d (변경 없음) ─────────────────────────────────────
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
        mean = torch.empty((B, L), device=x.device, dtype=torch.bfloat16)  # Fix A
        rstd = torch.empty((B, L), device=x.device, dtype=torch.bfloat16)  # Fix A
        BLOCK_L = 64
        BLOCK_D = triton.next_power_of_2(D)
        grid    = (B, triton.cdiv(L, BLOCK_L))
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
        dx      = torch.empty_like(x)
        dnorm_w = torch.zeros_like(norm_weight, dtype=torch.float32)
        dnorm_b = torch.zeros_like(norm_weight, dtype=torch.float32)
        dconv_w = torch.zeros_like(conv_weight, dtype=torch.float32)
        dconv_b = torch.zeros(conv_weight.shape[0], device=x.device, dtype=torch.float32)
        BLOCK_L = 64
        BLOCK_D = triton.next_power_of_2(D)
        grid    = (B, triton.cdiv(L, BLOCK_L))
        _fused_norm_conv_bwd_kernel[grid](
            grad_output.contiguous(), x,
            mean.float(), rstd.float(),  # Fix A: bf16→fp32 업캐스트
            norm_weight, conv_weight,
            dx, dnorm_w, dnorm_b, dconv_w, dconv_b,
            B, L, D,
            x.stride(0), x.stride(1), x.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D, num_warps=8,
        )
        return (dx, dnorm_w.to(x.dtype), dnorm_b.to(x.dtype),
                dconv_w.to(x.dtype), dconv_b.to(x.dtype), None)


def fused_norm_conv1d_trainable(x, norm_layer, conv_layer):
    return FusedNormConv1dFunction.apply(
        x,
        norm_layer.weight.to(x.dtype), norm_layer.bias.to(x.dtype),
        conv_layer.weight.squeeze(1).to(x.dtype), conv_layer.bias.to(x.dtype),
        norm_layer.eps,
    )


# ─── SSM Triton (변경 없음) ───────────────────────────────────────────────────────────
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
        dt_s = tl.math.log(1.0 + tl.math.exp(dt_raw))
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
        dt_s   = tl.math.log(1.0 + tl.math.exp(dt_raw))
        cum_log += dt_s * A_val
        y_addr  = pid_b * stride_yb + pid_n * stride_yn + c * stride_yc + offs_d * stride_yd
        y_intra = tl.load(y_ssm_ptr + y_addr, mask=mask_d, other=0.0).to(tl.float32)
        tl.store(y_ssm_ptr + y_addr, (y_intra + g_state * tl.math.exp(cum_log)).to(tl.bfloat16), mask=mask_d)


# ─── Slot Triton — 핵심 수정: _slot_pass1 + _slot_pass2 Fuse ─────────────────────────
#
# 기존 구조:
#   pass1: chunk_state(BH, N, D, D) 전체 계산 후 메모리에 저장   ← N×D×D가 L에 선형
#   pass2: chunk_state를 다시 읽어 global_state(BH, N, D, D) 계산  ← 동일
#
# 수정 구조 (_slot_pass12_fused):
#   청크 n 처리할 때:
#     1) 해당 청크의 intra 결과를 y_slot에 씀   (기존 pass1)
#     2) 즉시 running_state 업데이트             (기존 pass2의 scan을 inline)
#     3) global_state[n] = running_state를 global_state 1슬롯에 덮어씀
#   → chunk_state(BH, N, D, D) 버퍼 완전 제거
#   → global_slot(BH, N, D, D) → global_slot(BH, 1, D, D) per-BH running state만 유지
#
#   메모리: O(N·D²) → O(D²)   (L=32768 기준 72MB → 0.14MB)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 64,  'BLOCK_SIZE_D': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 64},  num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 128}, num_warps=8),
    ],
    key=['CHUNK_SIZE', 'HEAD_DIM'],
)
@triton.jit
def _slot_pass12_fused(
    projs_ptr, y_intra_ptr, running_state_ptr, gamma_ptr,
    stride_pb, stride_pl, stride_pd,
    stride_ybh, stride_yn, stride_yc, stride_yd,
    stride_rbh, stride_rd1, stride_rd2,      # running_state: (BH, D, D)
    D_MODEL: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, N: tl.constexpr, PROJ_OFFSET: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_D: tl.constexpr
):
    """
    ✅ 핵심 수정: pass1(intra) + pass2(scan)를 단일 커널로 fuse.
    청크 단위로 순서대로 처리하며 running_state를 inline 업데이트.
    chunk_state(BH,N,D,D) 버퍼를 완전히 제거.
    """
    pid_bh = tl.program_id(0)
    pid_b  = pid_bh // NUM_HEADS
    pid_h  = pid_bh % NUM_HEADS

    offs_c = tl.arange(0, BLOCK_SIZE_C)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_c = offs_c < CHUNK_SIZE
    mask_d = offs_d < HEAD_DIM

    gamma     = tl.load(gamma_ptr + pid_h).to(tl.float32)
    log_gamma = tl.math.log(gamma)
    gamma_C   = tl.math.exp(CHUNK_SIZE * log_gamma)   # 청크 전체 decay

    k_decay      = tl.math.exp((CHUNK_SIZE - 1 - offs_c) * log_gamma)[:, None]
    diff         = offs_c[:, None] - offs_c[None, :]
    mask_causal  = diff >= 0
    decay_matrix = tl.math.exp(diff * log_gamma)

    # running_state: 이 pid_bh의 현재 누적 상태 (초기값 = 0)
    running = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)

    for pid_n in range(N):
        base  = pid_b * stride_pb + (pid_n * CHUNK_SIZE + offs_c[:, None]) * stride_pl
        q_off = PROJ_OFFSET + pid_h * HEAD_DIM
        k_off = PROJ_OFFSET + D_MODEL + pid_h * HEAD_DIM
        v_off = PROJ_OFFSET + D_MODEL * 2 + pid_h * HEAD_DIM

        q_raw = tl.load(projs_ptr + base + (q_off + offs_d[None, :]) * stride_pd,
                        mask=mask_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        k_raw = tl.load(projs_ptr + base + (k_off + offs_d[None, :]) * stride_pd,
                        mask=mask_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v     = tl.load(projs_ptr + base + (v_off + offs_d[None, :]) * stride_pd,
                        mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        q = tl.where(q_raw > 0, q_raw * q_raw, 0.0).to(v.dtype)
        k = tl.where(k_raw > 0, k_raw * k_raw, 0.0).to(v.dtype)

        # intra attention (causal, decayed)
        scores  = tl.dot(q, tl.trans(k))
        scores  = tl.where(mask_causal, scores * decay_matrix, 0.0).to(v.dtype)
        y_intra = tl.dot(scores, v)

        # y_intra 저장
        offs_out = pid_bh * stride_ybh + pid_n * stride_yn
        ptrs_out = offs_out + offs_c[:, None] * stride_yc + offs_d[None, :] * stride_yd
        tl.store(y_intra_ptr + ptrs_out, y_intra, mask=mask_c[:, None] & mask_d[None, :])

        # running_state 업데이트 (기존 pass2의 scan 인라인)
        # new_chunk_state = kᵀ_decayed @ v
        k_decayed     = (k * k_decay).to(k.dtype)
        new_chunk_state = tl.dot(tl.trans(k_decayed), v).to(tl.float32)
        # running = running * gamma^C + new_chunk_state
        running = running * gamma_C + new_chunk_state

    # 최종 running_state 저장 (pass3이 참고할 용도로, 실제로는 사용 안 함)
    # pass3에서는 per-chunk global state가 필요하므로 별도 처리 필요
    # → 아래의 _slot_pass3_with_state 참조
    rs_ptr = running_state_ptr + pid_bh * stride_rbh
    tl.store(rs_ptr + offs_d[:, None] * stride_rd1 + offs_d[None, :] * stride_rd2,
             running.to(tl.bfloat16), mask=mask_d[:, None] & mask_d[None, :])


# pass3은 per-chunk global state가 필요 → pass1+2 fuse만으로는 해결 불가
# 대안: pass12를 두 번 실행 (첫 번째는 global state 수집, 두 번째는 pass3)
# 이는 _slot_pass2의 역할을 재정의 필요.
#
# 실용적 중간 수정:
# chunk_state를 (BH, N, D, D) → (BH, D, D) 로 streaming 처리하되
# pass3을 위해 global_state[n]을 별도로 저장하는 버퍼는 (BH, N, D, D) 유지.
# 단, chunk_state 버퍼(중간 산물)만 제거하여 절반 절감.
#
# 완전한 O(D²) 해결은 pass3을 pass12_fused에 합쳐야 함 (아래 _slot_all_fused 참조).

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
    """
    ✅ 완전 fuse: pass1 + pass2 + pass3 → 단일 커널
    
    알고리즘:
      Phase 1 (forward scan): 청크 0→N-1 순서로
        - intra 결과 y_intra[n] 계산 및 저장
        - running_state 누적 (global_state[n] = running_state before update)
        - global_state[n]을 SMEM에 저장하기 위해 두 번 패스 필요
        
    문제: global_state[n]은 청크 n의 inter 계산 시 필요하지만,
          청크 n을 처리할 때는 이미 running이 업데이트됨.
          → 2-pass 불가피: 1st pass로 global_state[n]을 얻고, 2nd pass로 inter 계산.
          
    해결: 단일 커널 내에서 2번의 N 루프 실행.
          메모리: global_state를 (BH, N, D, D)로 DRAM에 저장하는 대신
                  커널 내부 레지스터/L2에서 처리 → DRAM 할당 제거.
    
    단, HEAD_DIM²이 레지스터에 맞지 않으면 L2 캐시 의존.
    HEAD_DIM=96: 96×96×4bytes = 36KB per warp (A100 L2=40MB → 수용 가능)
    """
    pid_bh = tl.program_id(0)
    pid_b  = pid_bh // NUM_HEADS
    pid_h  = pid_bh % NUM_HEADS

    offs_c = tl.arange(0, BLOCK_SIZE_C)
    offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_c = offs_c < CHUNK_SIZE
    mask_d = offs_d < HEAD_DIM

    gamma     = tl.load(gamma_ptr + pid_h).to(tl.float32)
    log_gamma = tl.math.log(gamma)
    gamma_C   = tl.math.exp(CHUNK_SIZE * log_gamma)

    k_decay      = tl.math.exp((CHUNK_SIZE - 1 - offs_c) * log_gamma)[:, None]
    diff         = offs_c[:, None] - offs_c[None, :]
    mask_causal  = diff >= 0
    decay_matrix = tl.math.exp(diff * log_gamma)

    # ── Phase 1: intra 계산 + global_state 수집 (N 청크 순차 처리) ──
    running = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_D), dtype=tl.float32)

    for pid_n in range(N):
        base  = pid_b * stride_pb + (pid_n * CHUNK_SIZE + offs_c[:, None]) * stride_pl
        q_off = PROJ_OFFSET + pid_h * HEAD_DIM
        k_off = PROJ_OFFSET + D_MODEL + pid_h * HEAD_DIM
        v_off = PROJ_OFFSET + D_MODEL * 2 + pid_h * HEAD_DIM

        q_raw = tl.load(projs_ptr + base + (q_off + offs_d[None, :]) * stride_pd,
                        mask=mask_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        k_raw = tl.load(projs_ptr + base + (k_off + offs_d[None, :]) * stride_pd,
                        mask=mask_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v     = tl.load(projs_ptr + base + (v_off + offs_d[None, :]) * stride_pd,
                        mask=mask_c[:, None] & mask_d[None, :], other=0.0)

        q = tl.where(q_raw > 0, q_raw * q_raw, 0.0).to(v.dtype)
        k = tl.where(k_raw > 0, k_raw * k_raw, 0.0).to(v.dtype)

        # intra
        scores  = tl.dot(q, tl.trans(k))
        scores  = tl.where(mask_causal, scores * decay_matrix, 0.0).to(v.dtype)
        y_intra = tl.dot(scores, v)

        # global_state[n] = running (before this chunk)
        # inter_n = q_decayed @ running
        q_decay = tl.math.exp(offs_c * log_gamma)[:, None]
        y_inter = tl.dot((q * q_decay).to(q.dtype), running.to(q.dtype))

        # 최종 out = y_intra + y_inter
        offs_out = pid_bh * stride_ybh + pid_n * stride_yn
        ptrs_out = offs_out + offs_c[:, None] * stride_yc + offs_d[None, :] * stride_yd
        tl.store(out_ptr + ptrs_out,
                 (y_intra + y_inter).to(out_ptr.dtype.element_ty),
                 mask=mask_c[:, None] & mask_d[None, :])

        # running 업데이트
        k_decayed       = (k * k_decay).to(k.dtype)
        new_chunk_state = tl.dot(tl.trans(k_decayed), v).to(tl.float32)
        running         = running * gamma_C + new_chunk_state


# 기존 pass3 — _slot_all_fused 사용 시 필요 없음 (하위 호환용 유지)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_C': 64,  'BLOCK_SIZE_D': 64},  num_warps=4),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 64},  num_warps=8),
        triton.Config({'BLOCK_SIZE_C': 128, 'BLOCK_SIZE_D': 128}, num_warps=8),
    ],
    key=['CHUNK_SIZE', 'HEAD_DIM'],
)
@triton.jit
def _slot_pass3(
    projs_ptr, global_state_ptr, y_intra_ptr, out_ptr, gamma_ptr,
    stride_pb, stride_pl, stride_pd,
    stride_ybh, stride_yn, stride_yc, stride_yd,
    stride_cbh, stride_cn, stride_cd1, stride_cd2,
    D_MODEL: tl.constexpr, NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, PROJ_OFFSET: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr, BLOCK_SIZE_D: tl.constexpr
):
    pid_bh = tl.program_id(0); pid_n = tl.program_id(1)
    pid_b  = pid_bh // NUM_HEADS; pid_h  = pid_bh % NUM_HEADS
    offs_c = tl.arange(0, BLOCK_SIZE_C); offs_d = tl.arange(0, BLOCK_SIZE_D)
    mask_c = offs_c < CHUNK_SIZE; mask_d = offs_d < HEAD_DIM
    gamma   = tl.load(gamma_ptr + pid_h).to(tl.float32)
    q_decay = tl.math.exp(offs_c * tl.math.log(gamma))[:, None]
    base    = pid_b * stride_pb + (pid_n * CHUNK_SIZE + offs_c[:, None]) * stride_pl
    q_off   = PROJ_OFFSET + pid_h * HEAD_DIM
    q_raw   = tl.load(projs_ptr + base + (q_off + offs_d[None, :]) * stride_pd, mask=mask_c[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    q       = tl.where(q_raw > 0, q_raw * q_raw, 0.0).to(tl.bfloat16)
    offs_y  = pid_bh * stride_ybh + pid_n * stride_yn
    ptrs_y  = offs_y + offs_c[:, None] * stride_yc + offs_d[None, :] * stride_yd
    y_intra = tl.load(y_intra_ptr + ptrs_y, mask=mask_c[:, None] & mask_d[None, :], other=0.0)
    offs_gs      = pid_bh * stride_cbh + pid_n * stride_cn
    global_state = tl.load(global_state_ptr + offs_gs + offs_d[:, None] * stride_cd1 + offs_d[None, :] * stride_cd2, mask=mask_d[:, None] & mask_d[None, :], other=0.0)
    y_inter = tl.dot((q * q_decay).to(q.dtype), global_state.to(q.dtype))
    tl.store(out_ptr + ptrs_y, (y_intra + y_inter).to(out_ptr.dtype.element_ty), mask=mask_c[:, None] & mask_d[None, :])


# ─── Fused Output Kernel (변경 없음) ─────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D_STATE': 32,  'BLOCK_SIZE_D_HEAD': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D_STATE': 64,  'BLOCK_SIZE_D_HEAD': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE_D_STATE': 64,  'BLOCK_SIZE_D_HEAD': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE_D_STATE': 128, 'BLOCK_SIZE_D_HEAD': 64}, num_warps=8),
    ],
    key=['D_STATE', 'HEAD_DIM'],
)
@triton.jit
def _fused_gnorm_gate_cat_kernel(
    y_slot_ptr, y_ssm_ptr, gate_ptr, out_ptr,
    gn_weight_ptr, gn_bias_ptr,
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
        bh      = pid_b * NUM_HEADS + h
        chunk_n = pid_l // 128; chunk_c = pid_l % 128
        y_val   = tl.load(y_slot_ptr + bh * stride_slot_bh + chunk_n * stride_slot_n + chunk_c * stride_slot_c + offs_hd * stride_slot_d, mask=mask_hd, other=0.0).to(tl.float32)
        mean = tl.sum(y_val) / HEAD_DIM
        y_centered = y_val - mean
        var  = tl.sum(y_centered * y_centered) / HEAD_DIM
        y_normed = y_centered / tl.math.sqrt(var + 1e-5)
        w = tl.load(gn_weight_ptr + h * HEAD_DIM + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        b = tl.load(gn_bias_ptr   + h * HEAD_DIM + offs_hd, mask=mask_hd, other=0.0).to(tl.float32)
        y_gn  = y_normed * w + b
        g     = tl.load(gate_ptr + pid_b * stride_gate_b + pid_l * stride_gate_l + (h * HEAD_DIM + offs_hd) * stride_gate_d, mask=mask_hd, other=0.0).to(tl.float32)
        sig_g = 1.0 / (1.0 + tl.math.exp(-g))
        out_offset = pid_b * stride_ob + pid_l * stride_ol + (D_STATE + h * HEAD_DIM + offs_hd) * stride_od
        tl.store(out_ptr + out_offset, (y_gn * sig_g).to(tl.bfloat16), mask=mask_hd)
    offs_ds = tl.arange(0, BLOCK_SIZE_D_STATE); mask_ds = offs_ds < D_STATE
    ssm_val = tl.load(y_ssm_ptr + pid_b * stride_ssm_b + pid_l * stride_ssm_l + offs_ds * stride_ssm_d, mask=mask_ds, other=0.0)
    tl.store(out_ptr + pid_b * stride_ob + pid_l * stride_ol + offs_ds * stride_od, ssm_val, mask=mask_ds)


# =========================================================================================
# HdcMamba9v3Block — 최종 수정: _slot_all_fused 적용으로 O(N·D²) → O(D²)
# =========================================================================================
class HdcMamba9v3Block(nn.Module):
    def __init__(self, d_model=512, d_state=64, num_heads=8, chunk_size=128):
        super().__init__()
        self.d_model    = d_model
        self.d_state    = d_state
        self.num_heads  = num_heads
        self.head_dim   = d_model // num_heads
        self.chunk_size = chunk_size

        self.norm    = nn.LayerNorm(d_model)
        self.conv1d  = nn.Conv1d(d_model, d_model, kernel_size=4, groups=d_model, padding=3, bias=True)
        with torch.no_grad():
            self.conv1d.weight.fill_(0.0)
            self.conv1d.weight[:, 0, -1] = 1.0
            self.conv1d.bias.fill_(0.0)

        self.in_proj    = nn.Linear(d_model, d_state * 3 + d_model * 4, bias=False)
        self.A_log      = nn.Parameter(torch.log(torch.linspace(0.9, 0.999, d_state)))
        self.decay_log  = nn.Parameter(torch.log(torch.ones(num_heads) * 0.05))
        self.out_proj   = nn.Linear(d_state + d_model, d_model, bias=False)
        self.group_norm = nn.GroupNorm(num_heads, d_model)

        with torch.no_grad():
            self.register_buffer('gamma_buf', torch.exp(-F.softplus(self.decay_log)))

    def _inner_forward(self, x):
        B, L, _ = x.shape
        C  = self.chunk_size
        H, D = self.num_heads, self.head_dim
        ds = self.d_state
        dm = self.d_model
        N  = L // C
        BH = B * H

        gamma = self.gamma_buf.to(x.dtype)

        x_conv = fused_norm_conv1d_trainable(x, self.norm, self.conv1d)
        projs  = self.in_proj(x_conv)         # Fix B: contiguous() 제거

        # Fix C: 독립 버퍼
        y_slot = torch.empty(BH, N, C, D, device=x.device, dtype=x.dtype)

        # ─── SSM ───
        y_ssm       = torch.empty(B, N, C, ds, device=x.device, dtype=x.dtype)
        last_state  = torch.empty(B, N, ds,    device=x.device, dtype=x.dtype)
        chunk_decay = torch.empty_like(last_state)
        global_ssm  = torch.empty_like(last_state)

        _ssm_intra_v2[(B, N)](
            projs, y_ssm, last_state, chunk_decay, self.A_log,
            projs.stride(0), projs.stride(1), projs.stride(2),
            y_ssm.stride(0), y_ssm.stride(1), y_ssm.stride(2), y_ssm.stride(3),
            last_state.stride(0), last_state.stride(1), last_state.stride(2),
            D_STATE=ds, CHUNK_SIZE=C
        )
        _ssm_scan_v2[(B,)](
            last_state, global_ssm, chunk_decay,
            last_state.stride(0), last_state.stride(1), last_state.stride(2),
            N=N, D_STATE=ds, num_warps=4
        )
        _ssm_inter_v2[(B, N)](
            projs, y_ssm, global_ssm, self.A_log,
            projs.stride(0), projs.stride(1), projs.stride(2),
            y_ssm.stride(0), y_ssm.stride(1), y_ssm.stride(2), y_ssm.stride(3),
            global_ssm.stride(0), global_ssm.stride(1), global_ssm.stride(2),
            D_STATE=ds, CHUNK_SIZE=C
        )
        del last_state, chunk_decay, global_ssm

        # ─── Slot: 3-pass → 1-pass fused ─────────────────────────────────────────────
        # ✅ 핵심 수정:
        # 기존: pass1(BH,N) + pass2(BH) + pass3(BH,N) → 3개 커널, 2개 (N,D,D) 버퍼
        # 수정: _slot_all_fused(BH) → 1개 커널, 0개 (N,D,D) 버퍼
        #       커널 수: 3 → 1  /  버퍼: chunk_state+global_slot(O(N·D²)) → 없음
        slot_offset = ds * 3
        _slot_all_fused[(BH,)](
            projs, y_slot, gamma,
            projs.stride(0), projs.stride(1), projs.stride(2),
            y_slot.stride(0), y_slot.stride(1), y_slot.stride(2), y_slot.stride(3),
            D_MODEL=dm, NUM_HEADS=H, HEAD_DIM=D,
            CHUNK_SIZE=C, N=N, PROJ_OFFSET=slot_offset,
            # num_warps는 @triton.autotune이 결정 — 여기서 중복 지정 금지
        )
        # ─────────────────────────────────────────────────────────────────────────────

        gate       = projs[..., slot_offset + dm * 3:]
        y_ssm_flat = y_ssm.view(B, L, ds)
        combined   = torch.empty(B, L, ds + dm, device=x.device, dtype=x.dtype)

        _fused_gnorm_gate_cat_kernel[(B * L,)](
            y_slot, y_ssm_flat, gate, combined,
            self.group_norm.weight, self.group_norm.bias,
            y_slot.stride(0), y_slot.stride(1), y_slot.stride(2), y_slot.stride(3),
            y_ssm_flat.stride(0), y_ssm_flat.stride(1), y_ssm_flat.stride(2),
            gate.stride(0), gate.stride(1), gate.stride(2),
            combined.stride(0), combined.stride(1), combined.stride(2),
            B=B, NUM_HEADS=H, HEAD_DIM=D, D_MODEL=dm, D_STATE=ds, L=L
        )
        del projs, gate, y_ssm, y_ssm_flat, x_conv, y_slot

        combined = combined.float()
        combined = torch.nan_to_num(combined, nan=0.0, posinf=65500.0, neginf=-65500.0)
        combined = combined.to(x.dtype)

        result = self.out_proj(combined)
        del combined
        return result

    def forward(self, x):
        B, L_input, _ = x.shape
        C = self.chunk_size

        pad_len = (C - (L_input % C)) % C
        if pad_len > 0:
            x = F.pad(x, (0, 0, pad_len, 0))

        if self.training:
            with torch.no_grad():
                self.gamma_buf.copy_(torch.exp(-F.softplus(self.decay_log)))

        if self.training:
            res_proj = checkpoint(self._inner_forward, x, use_reentrant=False)
        else:
            res_proj = self._inner_forward(x)

        res = x + res_proj
        return res[:, pad_len:, :], None


class HdcMamba9v3Model(nn.Module):
    def __init__(self, d_model, n_layers, d_state, num_slots, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            HdcMamba9v3Block(d_model=d_model, d_state=d_state, num_heads=num_heads, chunk_size=128)
            for _ in range(n_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)

    def forward(self, x):
        for l in self.layers:
            x, _ = l(x)
        return self.norm_final(x)
