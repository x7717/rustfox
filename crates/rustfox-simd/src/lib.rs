//! rustfox-simd: byte-scanning primitives for browser hot paths.
//!
//! Gecko scans a huge amount of bytes per page load: the HTML tokenizer looks
//! for `<` in text and `>`/`=`/whitespace inside tags; the URL parser splits
//! on `:`, `/`, `?`, `#`; the HTTP header parser hunts `\n` and `:`.
//! memchr-class scanning is therefore the lowest-level lever in the whole
//! pipeline — every cycle saved here multiplies through every subsystem.
//!
//! Three tiers, selected at runtime:
//!
//! 1. **AVX2** — 32 bytes per compare (`_mm256_cmpeq_epi8` + movemask + ctz)
//! 2. **SSE2** — 16 bytes per compare (baseline on x86_64, always present)
//! 3. **SWAR** — 8 bytes per compare with pure integer ops ("SIMD within a
//!    register", the portable fallback for non-x86 targets)
//!
//! Every tier is property-tested against a naive byte loop, including the
//! nasty boundary lengths (0..=96) where the fixed-width tails live.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Runtime tier, detected once per process and cached.
///
/// `is_x86_feature_detected!` itself caches the CPUID result, but it still
/// costs an atomic load + branch per call — and the HTML tokenizer calls
/// `find_byte` ~27k times per page with 15–50 byte payloads. Detection is
/// therefore hoisted into a per-process `OnceLock` and hidden behind a
/// length-based fast path: short slices never touch it at all.
static TIER: std::sync::OnceLock<Tier> = std::sync::OnceLock::new();

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    #[cfg(not(target_arch = "x86_64"))]
    Scalar,
}

#[inline]
fn tier() -> Tier {
    *TIER.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                Tier::Avx2
            } else {
                Tier::Sse2
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Tier::Scalar
        }
    })
}

/// Index of the first `needle` byte in `hay`.
///
/// Hybrid dispatch by length: the SIMD prologue (detection, vector setup,
/// mask handling) costs more than the scan itself for short slices, so
/// inputs under 32 bytes take the scalar/SWAR path with zero setup.
#[inline]
pub fn find_byte(hay: &[u8], needle: u8) -> Option<usize> {
    if hay.len() >= 32 {
        match tier() {
            #[cfg(target_arch = "x86_64")]
            Tier::Avx2 => {
                // SAFETY: avx2 was detected at process start.
                unsafe { find_avx2(hay, needle) }
            }
            #[cfg(target_arch = "x86_64")]
            Tier::Sse2 => {
                // SAFETY: sse2 is a baseline feature of every x86_64 target.
                unsafe { find_sse2(hay, needle) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Tier::Scalar => find_swar(hay, needle),
        }
    } else if hay.len() >= 8 {
        find_swar(hay, needle)
    } else {
        hay.iter().position(|&b| b == needle)
    }
}

/// Index of the first byte equal to `a` **or** `b`.
///
/// Used by the URL parser (split authority on `/` or `?`) and the HTTP
/// parser (find line end or terminator) — one pass instead of two.
/// Same hybrid length dispatch as [`find_byte`].
#[inline]
pub fn find_any2(hay: &[u8], a: u8, b: u8) -> Option<usize> {
    if a == b {
        return find_byte(hay, a);
    }
    if hay.len() >= 32 {
        match tier() {
            #[cfg(target_arch = "x86_64")]
            Tier::Avx2 => {
                // SAFETY: avx2 was detected at process start.
                unsafe { any2_avx2(hay, a, b) }
            }
            #[cfg(target_arch = "x86_64")]
            Tier::Sse2 => {
                // SAFETY: sse2 is a baseline feature of every x86_64 target.
                unsafe { any2_sse2(hay, a, b) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            Tier::Scalar => any2_swar(hay, a, b),
        }
    } else if hay.len() >= 8 {
        any2_swar(hay, a, b)
    } else {
        hay.iter().position(|&c| c == a || c == b)
    }
}

/// `find_byte`, starting at `from` (index is still absolute).
#[inline]
pub fn find_byte_from(hay: &[u8], needle: u8, from: usize) -> Option<usize> {
    let start = from.min(hay.len());
    find_byte(&hay[start..], needle).map(|i| i + start)
}

/// `find_any2`, starting at `from` (index is still absolute).
#[inline]
pub fn find_any2_from(hay: &[u8], a: u8, b: u8, from: usize) -> Option<usize> {
    let start = from.min(hay.len());
    find_any2(&hay[start..], a, b).map(|i| i + start)
}

// ---------------------------------------------------------------------------
// x86_64: AVX2 (32 bytes/step)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_avx2(hay: &[u8], needle: u8) -> Option<usize> {
    let vn = _mm256_set1_epi8(needle as i8);
    let n = hay.len();
    let mut i = 0usize;
    while i + 32 <= n {
        let v = _mm256_loadu_si256(hay.as_ptr().add(i) as *const __m256i);
        let m = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vn)) as u32;
        if m != 0 {
            return Some(i + m.trailing_zeros() as usize);
        }
        i += 32;
    }
    find_sse2(&hay[i..], needle).map(|j| i + j)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn any2_avx2(hay: &[u8], a: u8, b: u8) -> Option<usize> {
    let va = _mm256_set1_epi8(a as i8);
    let vb = _mm256_set1_epi8(b as i8);
    let n = hay.len();
    let mut i = 0usize;
    while i + 32 <= n {
        let v = _mm256_loadu_si256(hay.as_ptr().add(i) as *const __m256i);
        let m = (_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, va))
            | _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, vb))) as u32;
        if m != 0 {
            return Some(i + m.trailing_zeros() as usize);
        }
        i += 32;
    }
    any2_sse2(&hay[i..], a, b).map(|j| i + j)
}

// ---------------------------------------------------------------------------
// x86_64: SSE2 (16 bytes/step, baseline)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn find_sse2(hay: &[u8], needle: u8) -> Option<usize> {
    let vn = _mm_set1_epi8(needle as i8);
    let n = hay.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v = _mm_loadu_si128(hay.as_ptr().add(i) as *const __m128i);
        let m = _mm_movemask_epi8(_mm_cmpeq_epi8(v, vn)) as u32;
        if m != 0 {
            return Some(i + m.trailing_zeros() as usize);
        }
        i += 16;
    }
    find_swar(&hay[i..], needle).map(|j| i + j)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn any2_sse2(hay: &[u8], a: u8, b: u8) -> Option<usize> {
    let va = _mm_set1_epi8(a as i8);
    let vb = _mm_set1_epi8(b as i8);
    let n = hay.len();
    let mut i = 0usize;
    while i + 16 <= n {
        let v = _mm_loadu_si128(hay.as_ptr().add(i) as *const __m128i);
        let m = (_mm_movemask_epi8(_mm_cmpeq_epi8(v, va))
            | _mm_movemask_epi8(_mm_cmpeq_epi8(v, vb))) as u32;
        if m != 0 {
            return Some(i + m.trailing_zeros() as usize);
        }
        i += 16;
    }
    any2_swar(&hay[i..], a, b).map(|j| i + j)
}

// ---------------------------------------------------------------------------
// SWAR fallback: 8 bytes/step, portable
// ---------------------------------------------------------------------------

const SWAR_LO: u64 = 0x0101_0101_0101_0101;
const SWAR_HI: u64 = 0x8080_8080_8080_8080;

/// Bit-trick: for `x = w ^ broadcast(needle)`, marks the high bit of every
/// zero byte lane. `trailing_zeros() >> 3` is the first matching lane.
#[inline]
fn swar_zero_mask(x: u64) -> u64 {
    x.wrapping_sub(SWAR_LO) & !x & SWAR_HI
}

/// SWAR fallback: 8 bytes/step, portable.
///
/// `#[inline]` matters here: the tokenizer calls this on 15–50 byte slices
/// ~27k times per page, and a cross-crate non-inlined call costs more than
/// the scan itself. Inlined, the loop fuses into the caller's state machine.
#[inline]
pub fn find_swar(hay: &[u8], needle: u8) -> Option<usize> {
    let target = (needle as u64).wrapping_mul(SWAR_LO);
    let (chunks, rest) = hay.as_chunks::<8>();
    for (i, chunk) in chunks.iter().enumerate() {
        let w = u64::from_le_bytes(*chunk);
        let m = swar_zero_mask(w ^ target);
        if m != 0 {
            return Some(i * 8 + (m.trailing_zeros() >> 3) as usize);
        }
    }
    let idx = chunks.len() * 8;
    for (k, &c) in rest.iter().enumerate() {
        if c == needle {
            return Some(idx + k);
        }
    }
    None
}
/// Same rationale as [`find_swar`] for the two-byte variant.
#[inline]
pub fn any2_swar(hay: &[u8], a: u8, b: u8) -> Option<usize> {
    let ta = (a as u64).wrapping_mul(SWAR_LO);
    let tb = (b as u64).wrapping_mul(SWAR_LO);
    let (chunks, rest) = hay.as_chunks::<8>();
    for (i, chunk) in chunks.iter().enumerate() {
        let w = u64::from_le_bytes(*chunk);
        // a and b differ (checked by the caller), so the two match sets are
        // disjoint per lane and the lowest marked lane is the first hit.
        let m = swar_zero_mask(w ^ ta) | swar_zero_mask(w ^ tb);
        if m != 0 {
            return Some(i * 8 + (m.trailing_zeros() >> 3) as usize);
        }
    }
    let idx = chunks.len() * 8;
    for (k, &c) in rest.iter().enumerate() {
        if c == a || c == b {
            return Some(idx + k);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Property tests: every tier must agree with a naive byte loop.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_find(hay: &[u8], n: u8) -> Option<usize> {
        hay.iter().position(|&b| b == n)
    }

    fn naive_any2(hay: &[u8], a: u8, b: u8) -> Option<usize> {
        hay.iter().position(|&c| c == a || c == b)
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
    }

    #[test]
    fn matches_naive_on_random_buffers() {
        let mut rng = Lcg(0xFEED_u64.wrapping_mul(0x2545F4914F6CDD1D) | 1);
        for _ in 0..3000 {
            let len = (rng.next() >> 33) as usize % 260;
            let hay: Vec<u8> = (0..len).map(|_| (rng.next() >> 40) as u8).collect();
            let needle = (rng.next() >> 40) as u8;
            assert_eq!(
                find_byte(&hay, needle),
                naive_find(&hay, needle),
                "find_byte len={len} needle={needle}"
            );
            let a = (rng.next() >> 40) as u8;
            let b = (rng.next() >> 40) as u8;
            assert_eq!(
                find_any2(&hay, a, b),
                naive_any2(&hay, a, b),
                "find_any2 len={len} a={a} b={b}"
            );
        }
    }

    #[test]
    fn boundary_lengths_sweep() {
        // Sweeps every length around the SIMD step boundaries (16/32/64),
        // where the fixed-width tail handling hides bugs.
        for len in 0..=100usize {
            let hay: Vec<u8> = (0..len).map(|i| (i * 7 + 13) as u8).collect();
            for &n in &[0u8, 13, 7, 20, 27, 90, 255] {
                assert_eq!(find_byte(&hay, n), naive_find(&hay, n), "len={len} n={n}");
                assert_eq!(
                    find_any2(&hay, n, 200),
                    naive_any2(&hay, n, 200),
                    "any2 len={len} n={n}"
                );
            }
        }
    }

    #[test]
    fn from_offset_agrees_with_slicing() {
        let hay = b"the quick brown fox jumps over the lazy dog";
        for from in 0..hay.len() {
            for &n in b"ozq\0" {
                let direct = find_byte_from(hay, n, from);
                let sliced = find_byte(&hay[from..], n).map(|i| i + from);
                assert_eq!(direct, sliced, "from={from} n={n}");
            }
        }
    }

    #[test]
    fn empty_and_single() {
        assert_eq!(find_byte(b"", b'x'), None);
        assert_eq!(find_any2(b"", b'x', b'y'), None);
        assert_eq!(find_byte(b"x", b'x'), Some(0));
        assert_eq!(find_any2(b"y", b'x', b'y'), Some(0));
    }
}
