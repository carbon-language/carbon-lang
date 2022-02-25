//===-- Elementary operations for x86 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_X86_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_X86_H

#include "src/__support/CPP/Bit.h"
#include "src/__support/architectures.h"

#if defined(LLVM_LIBC_ARCH_X86)

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t

#ifdef __SSE2__
#include <immintrin.h>
#endif //  __SSE2__

#include "src/string/memory_utils/elements.h" // __llvm_libc::scalar

// Fixed-size Vector Operations
// ----------------------------

namespace __llvm_libc {
namespace x86 {

#ifdef __SSE2__
template <typename Base> struct Vector : public Base {
  static void copy(char *__restrict dst, const char *__restrict src) {
    Base::store(dst, Base::load(src));
  }

  static void move(char *dst, const char *src) {
    Base::store(dst, Base::load(src));
  }

  static bool equals(const char *a, const char *b) {
    return Base::not_equal_mask(Base::load(a), Base::load(b)) == 0;
  }

  static int three_way_compare(const char *a, const char *b) {
    const auto mask = Base::not_equal_mask(Base::load(a), Base::load(b));
    if (!mask)
      return 0;
    return char_diff(a, b, mask);
  }

  static void splat_set(char *dst, const unsigned char value) {
    Base::store(dst, Base::get_splatted_value(value));
  }

  static int char_diff(const char *a, const char *b, uint64_t mask) {
    const size_t diff_index = __builtin_ctzll(mask);
    const int ca = (unsigned char)a[diff_index];
    const int cb = (unsigned char)b[diff_index];
    return ca - cb;
  }
};

struct M128 {
  static constexpr size_t SIZE = 16;
  using T = char __attribute__((__vector_size__(SIZE)));
  static uint16_t mask(T value) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return _mm_movemask_epi8(__llvm_libc::bit_cast<__m128i>(value));
  }
  static uint16_t not_equal_mask(T a, T b) { return mask(a != b); }
  static T load(const char *ptr) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return __llvm_libc::bit_cast<T>(
        _mm_loadu_si128(reinterpret_cast<__m128i_u const *>(ptr)));
  }
  static void store(char *ptr, T value) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return _mm_storeu_si128(reinterpret_cast<__m128i_u *>(ptr),
                            __llvm_libc::bit_cast<__m128i>(value));
  }
  static T get_splatted_value(const char v) {
    const T splatted = {v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
    return splatted;
  }
};

using Vector128 = Vector<M128>; // 16 Bytes

#ifdef __AVX2__
struct M256 {
  static constexpr size_t SIZE = 32;
  using T = char __attribute__((__vector_size__(SIZE)));
  static uint32_t mask(T value) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return _mm256_movemask_epi8(__llvm_libc::bit_cast<__m256i>(value));
  }
  static uint32_t not_equal_mask(T a, T b) { return mask(a != b); }
  static T load(const char *ptr) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return __llvm_libc::bit_cast<T>(
        _mm256_loadu_si256(reinterpret_cast<__m256i const *>(ptr)));
  }
  static void store(char *ptr, T value) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr),
                               __llvm_libc::bit_cast<__m256i>(value));
  }
  static T get_splatted_value(const char v) {
    const T splatted = {v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                        v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
    return splatted;
  }
};

using Vector256 = Vector<M256>; // 32 Bytes

#if defined(__AVX512F__) and defined(__AVX512BW__)
struct M512 {
  static constexpr size_t SIZE = 64;
  using T = char __attribute__((__vector_size__(SIZE)));
  static uint64_t not_equal_mask(T a, T b) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return _mm512_cmpneq_epi8_mask(__llvm_libc::bit_cast<__m512i>(a),
                                   __llvm_libc::bit_cast<__m512i>(b));
  }
  static T load(const char *ptr) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return __llvm_libc::bit_cast<T>(_mm512_loadu_epi8(ptr));
  }
  static void store(char *ptr, T value) {
    // NOLINTNEXTLINE(llvmlibc-callee-namespace)
    return _mm512_storeu_epi8(ptr, __llvm_libc::bit_cast<__m512i>(value));
  }
  static T get_splatted_value(const char v) {
    const T splatted = {v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                        v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                        v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v,
                        v, v, v, v, v, v, v, v, v, v, v, v, v, v, v, v};
    return splatted;
  }
};
using Vector512 = Vector<M512>;

#endif // defined(__AVX512F__) and defined(__AVX512BW__)
#endif // __AVX2__
#endif // __SSE2__

using _1 = __llvm_libc::scalar::_1;
using _2 = __llvm_libc::scalar::_2;
using _3 = __llvm_libc::scalar::_3;
using _4 = __llvm_libc::scalar::_4;
using _8 = __llvm_libc::scalar::_8;
#if defined(__AVX512F__) && defined(__AVX512BW__)
using _16 = __llvm_libc::x86::Vector128;
using _32 = __llvm_libc::x86::Vector256;
using _64 = __llvm_libc::x86::Vector512;
using _128 = __llvm_libc::Repeated<_64, 2>;
#elif defined(__AVX2__)
using _16 = __llvm_libc::x86::Vector128;
using _32 = __llvm_libc::x86::Vector256;
using _64 = __llvm_libc::Repeated<_32, 2>;
using _128 = __llvm_libc::Repeated<_32, 4>;
#elif defined(__SSE2__)
using _16 = __llvm_libc::x86::Vector128;
using _32 = __llvm_libc::Repeated<_16, 2>;
using _64 = __llvm_libc::Repeated<_16, 4>;
using _128 = __llvm_libc::Repeated<_16, 8>;
#else
using _16 = __llvm_libc::Repeated<_8, 2>;
using _32 = __llvm_libc::Repeated<_8, 4>;
using _64 = __llvm_libc::Repeated<_8, 8>;
using _128 = __llvm_libc::Repeated<_8, 16>;
#endif

struct Accelerator {
  static void copy(char *dst, const char *src, size_t count) {
    asm volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
  }
};

} // namespace x86
} // namespace __llvm_libc

#endif // defined(LLVM_LIBC_ARCH_X86)

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_ELEMENTS_X86_H
