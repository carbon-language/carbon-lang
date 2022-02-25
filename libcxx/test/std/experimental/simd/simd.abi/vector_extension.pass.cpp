//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <experimental/simd>
//
// [simd.abi]

#include <experimental/simd>
#include <cstdint>

#include "test_macros.h"

namespace ex = std::experimental::parallelism_v2;

constexpr inline int reg_width() {
#if defined(__AVX__)
  return 32;
#else
  return 16;
#endif
}

#ifndef _LIBCPP_HAS_NO_VECTOR_EXTENSION

static_assert(
    sizeof(ex::simd<char, ex::__simd_abi<ex::_StorageKind::_VecExt, 1>>) == 1,
    "");
static_assert(
    sizeof(ex::simd<char, ex::__simd_abi<ex::_StorageKind::_VecExt, 2>>) == 2,
    "");
static_assert(
    sizeof(ex::simd<char, ex::__simd_abi<ex::_StorageKind::_VecExt, 3>>) == 4,
    "");
static_assert(
    sizeof(ex::simd<char, ex::__simd_abi<ex::_StorageKind::_VecExt, 12>>) == 16,
    "");
static_assert(
    sizeof(ex::simd<int32_t, ex::__simd_abi<ex::_StorageKind::_VecExt, 3>>) ==
        16,
    "");
static_assert(
    sizeof(ex::simd<int32_t, ex::__simd_abi<ex::_StorageKind::_VecExt, 5>>) ==
        32,
    "");
static_assert(
    std::is_same<ex::simd_abi::native<int8_t>,
                 ex::__simd_abi<ex::_StorageKind::_VecExt, reg_width()>>::value,
    "");
#else
static_assert(
    std::is_same<ex::simd_abi::native<int8_t>,
                 ex::__simd_abi<ex::_StorageKind::_Array, reg_width()>>::value,
    "");

#endif

static_assert(std::is_same<ex::simd_abi::compatible<int8_t>,
                           ex::__simd_abi<ex::_StorageKind::_Array, 16>>::value,
              "");

int main(int, char**) {
  return 0;
}
