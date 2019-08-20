// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INT_DIVIDE_H_
#define INT_DIVIDE_H_

// Work around unoptimized implementations of unsigned integer division
// by constant values in some compilers (looking at YOU, clang 7!)

#ifdef __clang__
#if __clang_major__ < 8
#define USE_INT_DIVIDE_WORKAROUNDS 1
#endif
#endif

#include <cinttypes>

namespace Fortran::decimal {

template<typename UINT, UINT DENOM> inline constexpr UINT FastDivision(UINT n) {
  return n / DENOM;
}

#if USE_INT_DIVIDE_WORKAROUNDS
template<> inline constexpr std::uint64_t FastDivision<std::uint64_t, 10000000000000000u>(std::uint64_t n) {
  return (static_cast<__uint128_t>(0x39a5652fb1137857) * n) >> (64 + 51);
}

template<> inline constexpr std::uint64_t FastDivision<std::uint64_t, 100000000000000u>(std::uint64_t n) {
  return (static_cast<__uint128_t>(0xb424dc35095cd81) * n) >> (64 + 42);
}

template<> inline constexpr std::uint32_t FastDivision<std::uint32_t, 1000000u>(std::uint32_t n) {
  return (static_cast<std::uint64_t>(0x431bde83) * n) >> (32 + 18);
}

template<> inline constexpr std::uint32_t FastDivision<std::uint32_t, 10000u>(std::uint32_t n) {
  return (static_cast<std::uint64_t>(0xd1b71759) * n) >> (32 + 13);
}

template<> inline constexpr std::uint64_t FastDivision<std::uint64_t, 10u>(std::uint64_t n) {
  return (static_cast<__uint128_t>(0xcccccccccccccccd) * n) >> (64 + 3);
}

template<> inline constexpr std::uint32_t FastDivision<std::uint32_t, 10u>(std::uint32_t n) {
  return (static_cast<std::uint64_t>(0xcccccccd) * n) >> (32 + 3);
}

template<> inline constexpr std::uint64_t FastDivision<std::uint64_t, 5u>(std::uint64_t n) {
  return (static_cast<__uint128_t>(0xcccccccccccccccd) * n) >> (64 + 2);
}

template<> inline constexpr std::uint32_t FastDivision<std::uint32_t, 5u>(std::uint32_t n) {
  return (static_cast<std::uint64_t>(0xcccccccd) * n) >> (32 + 2);
}
#endif

static_assert(FastDivision<std::uint64_t, 10000000000000000u>(9999999999999999u) == 0);
static_assert(FastDivision<std::uint64_t, 10000000000000000u>(10000000000000000u) == 1);
static_assert(FastDivision<std::uint64_t, 100000000000000u>(99999999999999u) == 0);
static_assert(FastDivision<std::uint64_t, 100000000000000u>(100000000000000u) == 1);
static_assert(FastDivision<std::uint32_t, 1000000u>(999999u) == 0);
static_assert(FastDivision<std::uint32_t, 1000000u>(1000000u) == 1);
static_assert(FastDivision<std::uint64_t, 10>(18446744073709551615u) == 1844674407370955161u);
static_assert(FastDivision<std::uint32_t, 10>(4294967295u) == 429496729u);
static_assert(FastDivision<std::uint64_t, 5>(18446744073709551615u) == 3689348814741910323u);
static_assert(FastDivision<std::uint32_t, 5>(4294967295u) == 858993459u);
}
#endif
