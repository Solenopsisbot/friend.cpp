#ifndef PTQ1_0_GLSL
#define PTQ1_0_GLSL

// Shared PTQ1_0 trit accessors. Lives in its own header because two consumers need it
// from different include chains: dequant_funcs.glsl (mul_mat_vec, get_rows,
// copy_from_quant) and mul_mm.comp (via mul_mm_funcs.glsl), and mul_mm.comp does not
// include dequant_funcs.glsl. Duplicating it would leave two copies that must stay in
// step with the CPU codec in ggml-quants.c, where a divergence shows up as wrong
// matmul results rather than a build error.
//
// Element order is not positional: a 16-byte chunk of qs where byte j carries
// elements t*16+j, then an 8-byte chunk carrying 80 + t*8 + (j-16), then qh at four
// trits per byte carrying 120 + t*2 + h. Trit t of a byte b is the CPU codec's
// q = (b * 3^t) mod 256, trit = (q * 3) >> 8 (then minus one).

// friend.cpp: the t-th trit in closed form (one multiply by 3^t) instead of stepping the
// base-3 recurrence t times; identical values. 3^0..3^3 are the bytes of 0x1B090301.
uint ptq1_0_pow3(uint t) {
    return t < 4u ? (0x1B090301u >> (8u * t)) & 0xFFu : 81u;
}

float ptq1_0_trit_of(uint b, uint pow3) {
    return float(int((((b * pow3) & 0xFFu) * 3u) >> 8u) - 1);
}

float ptq1_0_trit(uint ib, uint a_offset, uint e) {
    uint b;
    uint n;
    if (e < 80u) {
        b = uint(data_a[a_offset + ib].qs[e & 15u]);
        n = e >> 4u;
    } else if (e < 120u) {
        const uint t = e - 80u;
        b = uint(data_a[a_offset + ib].qs[16u + (t & 7u)]);
        n = t >> 3u;
    } else {
        const uint t = e - 120u;
        b = uint(data_a[a_offset + ib].qh[t & 1u]);
        n = t >> 1u;
    }
    return ptq1_0_trit_of(b, ptq1_0_pow3(n));
}

// friend.cpp: four consecutive elements e..e+3 (e % 4 == 0) from one 32-bit load, without
// branches. Below element 120 every such group is four consecutive qs bytes at one trit index
// n; the last group of each half of the qh tail is (qh0, n), (qh1, n), (qh0, n+1), (qh1, n+1)
// with n = (e - 120) / 2, handled by duplicating the two qh bytes and using a second power for
// the upper pair. Lanes of a mat-vec warp sit at different e, so the index maths is done with
// selects rather than if/else to keep the warp converged. The first Vulkan port decoded each
// element separately (byte fetch, branch, a loop of up to four steps), which made the PTQ1_0
// mat-vec ~17x slower than Q1_0's on a GTX 970; the values are unchanged.
// Needs data_a_packed32 (A_TYPE_PACKED32 = block_ptq1_0_packed32) in scope.
vec4 ptq1_0_trits4(uint ib, uint a_offset, uint e) {
    const bool lo   = e < 80u;
    const bool tail = e >= 120u;
    const uint t    = e - 80u;

    const uint widx = lo ? ((e & 15u) >> 2u) : (tail ? 6u : 4u + ((t & 7u) >> 2u));
    const uint n    = lo ? (e >> 4u)         : (tail ? ((e - 120u) >> 1u) : (t >> 3u));

    uint w = data_a_packed32[a_offset + ib].w[widx];
    w = tail ? (w & 0xFFFFu) * 0x00010001u : w; // qh0 qh1 qh0 qh1

    const uint p0 = ptq1_0_pow3(n);
    const uint p1 = tail ? ptq1_0_pow3(n + 1u) : p0;

    return vec4(ptq1_0_trit_of( w         & 0xFFu, p0),
                ptq1_0_trit_of((w >>  8u) & 0xFFu, p0),
                ptq1_0_trit_of((w >> 16u) & 0xFFu, p1),
                ptq1_0_trit_of( w >> 24u,          p1));
}

#endif // PTQ1_0_GLSL
