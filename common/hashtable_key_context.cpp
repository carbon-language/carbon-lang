// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashtable_key_context.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "llvm/Support/Compiler.h"

// Whether short comparisons can read past the end of the inputs, as long as
// the reads stay within a page that contains bytes of the inputs. Sanitizers
// would report those reads as out of bounds, or as races with writes to the
// bytes after the inputs. So would Valgrind and Arm memory tagging, which
// can't be detected here. The bytes past the end are masked off assuming a
// little-endian byte order.
#if !LLVM_ADDRESS_SANITIZER_BUILD && !LLVM_HWADDRESS_SANITIZER_BUILD && \
    !LLVM_MEMORY_SANITIZER_BUILD && !LLVM_THREAD_SANITIZER_BUILD &&     \
    __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define CARBON_STRING_BYTES_NE_OVERREAD 1
#endif

// Detect the architectures with dedicated implementations of the size classes
// over 16 bytes, and include the intrinsics they use.
//
// Reference documentation for these intrinsics:
// - https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
// - https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define CARBON_STRING_BYTES_NE_AARCH64 1
#elif defined(__SSE4_1__)
#include <x86intrin.h>
#define CARBON_STRING_BYTES_NE_X86 1
#endif

namespace Carbon::InternalHashtableEqDispatch {

// Loads a `T` from `bytes`, which don't need to be aligned.
template <typename T>
static auto Load(const std::byte* bytes) -> T {
  T value;
  std::memcpy(&value, bytes, sizeof(T));
  return value;
}

// Returns a `T` with bits set wherever the `T` values at `lhs` and `rhs`
// differ. This is zero when the values are equal and non-zero otherwise, and
// multiple results can be combined with `|`.
template <typename T>
static auto XorBytes(const std::byte* lhs, const std::byte* rhs) -> T {
  return Load<T>(lhs) ^ Load<T>(rhs);
}

// Reads `size` bytes for `size` in 1..8 into a `uint64_t`, reading no further
// than those bytes.
static auto LoadExact1To8(const std::byte* bytes, size_t size) -> uint64_t {
  if (size >= 4) {
    return Hasher::Read4To8(bytes, size);
  }
  return Hasher::Read1To3(bytes, size);
}

#if CARBON_STRING_BYTES_NE_OVERREAD

// Returns whether an 8-byte forward read from `bytes` stays within its 4 KiB
// page. Pages are at least 4 KiB on all supported platforms.
//
// Testing `((bytes + 8) & 0xff8) != 0` compiles to `lea` and `test` per
// pointer on x86-64. It rejects page offsets 4088-4095: one offset (4088) more
// than the unaligned offsets 4089-4095 that actually cross the page boundary,
// and never a page start (offset 0).
static auto Read8StaysInPage(const std::byte* bytes) -> bool {
  return ((reinterpret_cast<uintptr_t>(bytes) + 8) & 0xff8) != 0;
}

#if !defined(__BMI2__)
// Little-endian masks keeping the first `size` bytes of an 8-byte load, indexed
// directly by `size` for `size` in 1..8.
//
// Element 0 is unused padding (and omitted from the loop because shifting a
// 64-bit integer by 64 bits is undefined) so lookups can index by `size` rather
// than `size - 1`: on AArch64, `ldr` can't combine a scaled register index with
// a byte offset, so `size - 1` would cost an extra `add`.
alignas(64) static constexpr std::array<uint64_t, 9> SizeMasks8 = [] {
  std::array<uint64_t, 9> masks = {};
  for (size_t size = 1; size <= 8; ++size) {
    masks[size] = ~uint64_t{0} >> (8 * (8 - size));
  }
  return masks;
}();
#endif

#endif  // CARBON_STRING_BYTES_NE_OVERREAD

////////////////////////////////////////////////////////////////////////////////
//
// Size-specific byte comparisons (`BytesNe*`) and `StringBytesNe`.
//
// `StringBytesNe` and each `BytesNe*` function below compare bytes at `lhs` and
// `rhs` for the size or size range in the function's name. They return zero
// when all of the bytes are equal and some non-zero 64-bit value otherwise,
// rather than a `bool` or `bcmp`'s `int`. Callers may only compare the result
// with zero, which leaves each size class free to return whatever non-zero
// value is cheapest to produce. Callers typically branch on the result, so
// reducing it to a flag here, only for them to test that flag again, adds an op
// and a cycle before their branch resolves. Callers that need a `bool` value
// pay that cycle instead.
//
////////////////////////////////////////////////////////////////////////////////

// Compares `size` bytes for `size` in 1..8, reading no further than those
// bytes.
[[clang::noinline]] CARBON_STRING_BYTES_NE_CC static auto BytesNeExact1To8(
    const std::byte* lhs, const std::byte* rhs, size_t size) -> uint64_t {
  return LoadExact1To8(lhs, size) ^ LoadExact1To8(rhs, size);
}

#if CARBON_STRING_BYTES_NE_OVERREAD

// Compares `size` bytes for `size` in 1..8 without branching on `size`, by
// reading 8 bytes forward from each input, `xor`ing them, and masking off the
// bytes past `size` (`bzhi` with BMI2, otherwise `SizeMasks8`). Requires those
// reads to stay within a page.
//
// Handling these sizes without branches removes the 1-3 vs 4-8 branch, which
// mispredicts when short sizes vary.
//
// Reading forward from `lhs` and `rhs` is preferred over a backward read of
// the 8 bytes ending at `lhs + size` and `rhs + size`. A backward read's page
// check needs only a single `test` per pointer, but its load address depends
// on `size` (adding a cycle of address-generation latency) and inputs in the
// first 8 bytes of a page—including the start of any page-aligned buffer—can't
// use it.
static auto BytesNeOverread1To8(const std::byte* lhs, const std::byte* rhs,
                                size_t size) -> uint64_t {
  auto ne_bits = XorBytes<uint64_t>(lhs, rhs);
#if defined(__BMI2__)
  ne_bits = _bzhi_u64(ne_bits, 8 * size);
  // Keep `size` in its register past the computation of `8 * size`. Otherwise
  // LLVM computes that in place, and then `StringBytesNe` has to save and
  // restore the register on every call to preserve it for the caller. With
  // this, it goes in the scratch register instead.
  __asm__("" : : "r"(size));
  return ne_bits;
#else
  return ne_bits & SizeMasks8[size];
#endif
}

#endif  // CARBON_STRING_BYTES_NE_OVERREAD

// Exits on a mismatch in the first 8 bytes rather than combining it with the
// last 8 bytes with `|`. That leaves a single `xor` between the last loads and
// the result, and x86-64 fuses the first compare with a load and with the
// branch. The branch is only taken for inputs that differ, where the caller's
// branch on the result is no easier to predict.
static auto BytesNe9To16(const std::byte* lhs, const std::byte* rhs,
                         size_t size) -> uint64_t {
  if (LLVM_UNLIKELY(Load<uint64_t>(lhs) != Load<uint64_t>(rhs))) {
    return 1;
  }
  size_t tail_offset = size - sizeof(uint64_t);
  return XorBytes<uint64_t>(lhs + tail_offset, rhs + tail_offset);
}

#if CARBON_STRING_BYTES_NE_AARCH64

// Comparing `__uint128_t` keeps the loads and compares in general registers
// (`ldp` + `cmp` + `ccmp`), avoiding the latency of moving a result from Neon
// to general registers; LLVM would vectorize 64-bit comparisons into Neon.
//
// Combining the two halves with `|` rather than `||` loads both halves
// unconditionally and folds all four 64-bit compares into a single branchless
// `cmp` and `ccmp` chain, rather than splitting them across an early-exit
// branch.
//
// The tails are addressed from the ends of the inputs, rather than with a
// shared `size - 16` offset. LLVM computes that offset as part of the
// comparison of `size` that leads here, and a 16-byte load into general
// registers (`ldp`) can't add a register offset to its address, so each tail
// load would then wait an extra cycle for a second addition.
static auto BytesNe17To32(const std::byte* lhs, const std::byte* rhs,
                          size_t size) -> uint64_t {
  return (XorBytes<__uint128_t>(lhs, rhs) |
          XorBytes<__uint128_t>(lhs + size - 16, rhs + size - 16)) != 0;
}

// `BytesNe33To64` and `BytesNeOver64` pair two 32-byte comparisons:
// - `BytesNe32Neon` for predicted early-exit checks, where Neon's higher
//   throughput wins and its Neon-to-general-register latency is hidden by the
//   next check.
// - `BytesNe32` when producing the final return value, where `__uint128_t`
//   keeps the comparison in general registers (`ldp` + `cmp` + `ccmp`) to avoid
//   that Neon-to-general-register latency (LLVM would vectorize 64-bit
//   comparisons).

static auto BytesNe32(const std::byte* lhs, const std::byte* rhs) -> uint64_t {
  return (XorBytes<__uint128_t>(lhs, rhs) |
          XorBytes<__uint128_t>(lhs + 16, rhs + 16)) != 0;
}

static auto BytesNe32Neon(const std::byte* lhs, const std::byte* rhs)
    -> uint64_t {
  uint8x16_t eq_bytes =
      vceqq_u8(Load<uint8x16_t>(lhs), Load<uint8x16_t>(rhs)) &
      vceqq_u8(Load<uint8x16_t>(lhs + 16), Load<uint8x16_t>(rhs + 16));
  // The pairwise minimum of the 32-bit lanes has all of its low 64 bits set
  // exactly when all the bytes are equal.
  uint32x4_t lanes = vreinterpretq_u32_u8(eq_bytes);
  uint32x4_t min_lanes = vpminq_u32(lanes, lanes);
  return ~vgetq_lane_u64(vreinterpretq_u64_u32(min_lanes), 0);
}

static auto BytesNe33To64(const std::byte* lhs, const std::byte* rhs,
                          size_t size) -> uint64_t {
  if (BytesNe32Neon(lhs, rhs)) {
    return 1;
  }
  return BytesNe32(lhs + size - 32, rhs + size - 32);
}

[[clang::noinline]] CARBON_STRING_BYTES_NE_CC static auto BytesNeOver64(
    const std::byte* lhs, const std::byte* rhs, size_t size) -> uint64_t {
  // Compare 32 bytes at a time, finishing with the last 32 bytes, which can
  // overlap bytes that were already compared.
  size_t last = size - 32;
  for (size_t i = 0; i < last; i += 32) {
    if (BytesNe32Neon(lhs + i, rhs + i)) {
      return 1;
    }
  }
  return BytesNe32(lhs + last, rhs + last);
}

#elif CARBON_STRING_BYTES_NE_X86

// `BytesNe17To32` and `BytesNe33To64` compare bytes with `pcmpeqb` and extract
// a bitmask with `pmovmskb`, whereas reducing `XorBytes` with `ptest`
// (`_mm*_testz_si*`) below also needs `setcc` to materialize the return value.
// On Zen 5 that gets the result to the caller's branch 2 to 3 cycles sooner,
// without adding ops.
//
// TODO: Switch to scalar code once LLVM vectorizes it well. Today it builds a
// 256-bit compare with `vinserti128` for 17-32 bytes on x86-64-v3.

static auto BytesNe17To32(const std::byte* lhs, const std::byte* rhs,
                          size_t size) -> uint64_t {
  __m128i eq = _mm_cmpeq_epi8(Load<__m128i>(lhs), Load<__m128i>(rhs)) &
               _mm_cmpeq_epi8(Load<__m128i>(lhs + size - 16),
                              Load<__m128i>(rhs + size - 16));
  return static_cast<uint32_t>(_mm_movemask_epi8(eq)) ^ 0xffffU;
}

static auto BytesNe33To64(const std::byte* lhs, const std::byte* rhs,
                          size_t size) -> uint64_t {
#if defined(__AVX2__)
  __m256i eq = _mm256_cmpeq_epi8(Load<__m256i>(lhs), Load<__m256i>(rhs)) &
               _mm256_cmpeq_epi8(Load<__m256i>(lhs + size - 32),
                                 Load<__m256i>(rhs + size - 32));
  return ~static_cast<uint32_t>(_mm256_movemask_epi8(eq));
#else
  __m128i eq =
      _mm_cmpeq_epi8(Load<__m128i>(lhs), Load<__m128i>(rhs)) &
      _mm_cmpeq_epi8(Load<__m128i>(lhs + 16), Load<__m128i>(rhs + 16)) &
      _mm_cmpeq_epi8(Load<__m128i>(lhs + size - 32),
                     Load<__m128i>(rhs + size - 32)) &
      _mm_cmpeq_epi8(Load<__m128i>(lhs + size - 16),
                     Load<__m128i>(rhs + size - 16));
  return static_cast<uint32_t>(_mm_movemask_epi8(eq)) ^ 0xffffU;
#endif
}

// `BytesNeOver64` takes its result from the first 32 bytes, and checks the
// other bytes with predicted branches. Loading the first 32 bytes before the
// others keeps the other loads from delaying the result. Every size also
// compares the next 32 bytes, so loading them before branching on the size
// keeps a mispredicted branch from delaying them.
//
// `BytesNeOver64` uses `XorBytes` and `ptest` (`_mm*_testz_si*`) rather than
// `pcmpeqb` and `pmovmskb`, even though `pmovmskb` would have 2 to 4 cycles
// lower latency on Zen 5: with AVX2, LLVM replaces the last branch with a
// `cmov` when returning a `pmovmskb` mask, so that the result waits for every
// byte; without AVX2, it hoists and shares the first compare with
// `BytesNe33To64`, costing 2 cycles on that path.
#if defined(__AVX2__)
[[clang::noinline]] CARBON_STRING_BYTES_NE_CC static auto BytesNeOver64(
    const std::byte* lhs, const std::byte* rhs, size_t size) -> uint64_t {
  auto head_xor = XorBytes<__m256i>(lhs, rhs);
  auto second_xor = XorBytes<__m256i>(lhs + 32, rhs + 32);
  if (size <= 128) {
    __m256i rest_xor = second_xor |
                       XorBytes<__m256i>(lhs + size - 64, rhs + size - 64) |
                       XorBytes<__m256i>(lhs + size - 32, rhs + size - 32);
    if (!_mm256_testz_si256(rest_xor, rest_xor)) {
      return 1;
    }
    return !_mm256_testz_si256(head_xor, head_xor);
  }
  // Larger sizes finish by comparing bytes 64 to 128 and the last 128 bytes,
  // which can overlap bytes that were already compared. Over 256 bytes, they
  // first compare the bytes in between.
  auto xor_128 = [](const std::byte* a, const std::byte* b) -> __m256i {
    return XorBytes<__m256i>(a, b) | XorBytes<__m256i>(a + 32, b + 32) |
           XorBytes<__m256i>(a + 64, b + 64) |
           XorBytes<__m256i>(a + 96, b + 96);
  };
  size_t last = size - 128;
  if (size > 256) {
    // Compare 128 bytes at a time from where `lhs` is 32-byte aligned, so that
    // its loads don't cross cache lines.
    for (size_t i = 128 - reinterpret_cast<uintptr_t>(lhs) % 32; i < last;
         i += 128) {
      __m256i block_xor = xor_128(lhs + i, rhs + i);
      if (!_mm256_testz_si256(block_xor, block_xor)) {
        return 1;
      }
    }
  }
  __m256i rest_xor = second_xor | XorBytes<__m256i>(lhs + 64, rhs + 64) |
                     XorBytes<__m256i>(lhs + 96, rhs + 96) |
                     xor_128(lhs + last, rhs + last);
  if (!_mm256_testz_si256(rest_xor, rest_xor)) {
    return 1;
  }
  return !_mm256_testz_si256(head_xor, head_xor);
}
#else
// Without AVX2, each 32-byte comparison takes twice the instructions. To keep
// the code small, this has a single loop of 128-byte blocks, and every size
// finishes with the same check of the last 64 bytes.
[[clang::noinline]] CARBON_STRING_BYTES_NE_CC static auto BytesNeOver64(
    const std::byte* lhs, const std::byte* rhs, size_t size) -> uint64_t {
  auto xor_32 = [](const std::byte* a, const std::byte* b) -> __m128i {
    return XorBytes<__m128i>(a, b) | XorBytes<__m128i>(a + 16, b + 16);
  };
  __m128i head_xor = xor_32(lhs, rhs);
  __m128i second_xor = xor_32(lhs + 32, rhs + 32);
  if (size > 128) {
    // Compare 128-byte blocks from byte 64 until they reach the last 64 bytes,
    // moving a block back if it would pass the end. That's exactly one block
    // for sizes up to 256 bytes.
    //
    // Align the loop to a cache line so that its body spans four cache lines
    // rather than five.
    size_t i = 64;
    [[clang::code_align(64)]] do {
      size_t b = std::min(i, size - 128);
      __m128i block_xor = xor_32(lhs + b, rhs + b) |
                          xor_32(lhs + b + 32, rhs + b + 32) |
                          xor_32(lhs + b + 64, rhs + b + 64) |
                          xor_32(lhs + b + 96, rhs + b + 96);
      if (!_mm_testz_si128(block_xor, block_xor)) {
        return 1;
      }
      i += 128;
    } while (i < size - 64);
  }
  __m128i rest_xor = second_xor | xor_32(lhs + size - 64, rhs + size - 64) |
                     xor_32(lhs + size - 32, rhs + size - 32);
  if (!_mm_testz_si128(rest_xor, rest_xor)) {
    return 1;
  }
  return !_mm_testz_si128(head_xor, head_xor);
}
#endif  // defined(__AVX2__)

#else

// Rely on `memcmp` for other architectures, hoping for a dynamically selected
// implementation.
[[clang::noinline]] CARBON_STRING_BYTES_NE_CC static auto BytesNeOver16(
    const std::byte* lhs, const std::byte* rhs, size_t size) -> uint64_t {
  return std::memcmp(lhs, rhs, size) != 0;
}

#endif

// This is faster than available `memcmp` and `bcmp` implementations, even LLVM
// libc's, especially for the short sizes that dominate most string comparisons
// in practice.
//
// Dispatch branches between power-of-two size classes rather than using an
// exact-size jump table, which mispredicts much more often when short sizes
// vary.
//
// Everything uses plain unmasked loads with no AVX-512 masked loads or IFUNC
// dispatch:
// - Masked loads add about 10 cycles of latency up to 16 bytes.
// - On Zen 5, a masked load with a zero mask at an invalid address costs about
//   450 cycles in fault-suppression assist overhead, and one whose masked-off
//   bytes reach an unmapped page costs about 600 cycles.
// - An IFUNC adds a PLT hop to every call it covers.
//
// The function is aligned to a 64-byte cache line because where it starts
// within one can change its performance by several cycles on some CPUs, and the
// 1-8 fast path fits within that first cache line.
//
// Calls preserve most of the caller's registers, see
// `CARBON_STRING_BYTES_NE_CC`, so callers don't spill their own state around
// each comparison. In exchange, this function has to save any general register
// it uses beyond the result's and, on x86-64, a single scratch register. It
// never does: sizes up to 64 only need those and vector registers, and the rest
// are tail calls to functions (`BytesNeExact1To8` and `BytesNeOver64`) that
// save what they use when they run.
//
// TODO: Contribute this to LLVM libc's `bcmp`.
[[clang::noinline, gnu::aligned(64)]] CARBON_STRING_BYTES_NE_CC auto
StringBytesNe(const std::byte* lhs, const std::byte* rhs, size_t size)
    -> uint64_t {
  if (size <= 8) {
    // Size 0 exits before any read: empty inputs can have null or dangling
    // pointers, and must not read memory.
    if (size == 0) {
      return 0;
    }
#if CARBON_STRING_BYTES_NE_OVERREAD
    // Folding the `size == 0` exit and page checks into size-indexed tables
    // takes fewer instructions, but adds load latency to the check and, on
    // Zen 5, a fused load-op holds its integer ALU scheduler slot through the
    // load.
    if (LLVM_LIKELY(Read8StaysInPage(lhs) & Read8StaysInPage(rhs))) {
      return BytesNeOverread1To8(lhs, rhs, size);
    }
#endif
    [[clang::musttail]] return BytesNeExact1To8(lhs, rhs, size);
  }
  if (size <= 16) {
    return BytesNe9To16(lhs, rhs, size);
  }
#if CARBON_STRING_BYTES_NE_AARCH64 || CARBON_STRING_BYTES_NE_X86
  if (size <= 32) {
    return BytesNe17To32(lhs, rhs, size);
  }
  if (size <= 64) {
    return BytesNe33To64(lhs, rhs, size);
  }
  [[clang::musttail]] return BytesNeOver64(lhs, rhs, size);
#else
  [[clang::musttail]] return BytesNeOver16(lhs, rhs, size);
#endif
}

}  // namespace Carbon::InternalHashtableEqDispatch
