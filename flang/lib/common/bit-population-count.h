// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_COMMON_BIT_POPULATION_COUNT_H_
#define FORTRAN_COMMON_BIT_POPULATION_COUNT_H_

// Fast and portable functions that implement Fortran's POPCNT and POPPAR
// intrinsic functions.  POPCNT returns the number of bits that are set (1)
// in its argument.  POPPAR is a parity function that returns true
// when POPCNT is odd.

#include <cinttypes>

namespace Fortran::common {

inline constexpr int BitPopulationCount(std::uint64_t x) {
  // In each of the 32 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555);
  // Combine into 16 4-bit fields, each holding [0..4]
  x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
  // Now 8 8-bit fields, each with [0..8] in their lower 4 bits.
  x = (x & 0x0f0f0f0f0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f0f0f0f0f);
  // Now 4 16-bit fields, each with [0..16] in their lower 5 bits.
  x = (x & 0x001f001f001f001f) + ((x >> 8) & 0x001f001f001f001f);
  // Now 2 32-bit fields, each with [0..32] in their lower 6 bits.
  x = (x & 0x0000003f0000003f) + ((x >> 16) & 0x0000003f0000003f);
  // Last step: 1 64-bit field, with [0..64]
  return (x & 0x7f) + (x >> 32);
}

inline constexpr int BitPopulationCount(std::uint32_t x) {
  // In each of the 16 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x55555555) + ((x >> 1) & 0x55555555);
  // Combine into 8 4-bit fields, each holding [0..4]
  x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
  // Now 4 8-bit fields, each with [0..8] in their lower 4 bits.
  x = (x & 0x0f0f0f0f) + ((x >> 4) & 0x0f0f0f0f);
  // Now 2 16-bit fields, each with [0..16] in their lower 5 bits.
  x = (x & 0x001f001f) + ((x >> 8) & 0x001f001f);
  // Last step: 1 32-bit field, with [0..32]
  return (x & 0x3f) + (x >> 16);
}

inline constexpr int BitPopulationCount(std::uint16_t x) {
  // In each of the 8 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x5555) + ((x >> 1) & 0x5555);
  // Combine into 4 4-bit fields, each holding [0..4]
  x = (x & 0x3333) + ((x >> 2) & 0x3333);
  // Now 2 8-bit fields, each with [0..8] in their lower 4 bits.
  x = (x & 0x0f0f) + ((x >> 4) & 0x0f0f);
  // Last step: 1 16-bit field, with [0..16]
  return (x & 0x1f) + (x >> 8);
}

inline constexpr int BitPopulationCount(std::uint8_t x) {
  // In each of the 4 2-bit fields, count the bits that were present.
  // This leaves a value [0..2] in each of these 2-bit fields.
  x = (x & 0x55) + ((x >> 1) & 0x55);
  // Combine into 2 4-bit fields, each holding [0..4]
  x = (x & 0x33) + ((x >> 2) & 0x33);
  // Last step: 1 8-bit field, with [0..8]
  return (x & 0xf) + (x >> 4);
}

template<typename UINT> inline constexpr bool Parity(UINT x) {
  return BitPopulationCount(x) & 1;
}

// "Parity is for farmers." -- Seymour R. Cray

template<typename UINT> inline constexpr int TrailingZeroCount(UINT x) {
  return BitPopulationCount(x ^ (x - 1)) - !x;
}
}  // namespace Fortran::common
#endif  // FORTRAN_COMMON_BIT_POPULATION_COUNT_H_
