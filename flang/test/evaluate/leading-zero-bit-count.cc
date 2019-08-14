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

#include "../../lib/common/leading-zero-bit-count.h"
#include "testing.h"

using Fortran::common::LeadingZeroBitCount;

int main() {
  MATCH(64, LeadingZeroBitCount(std::uint64_t{0}));
  for (int j{0}; j < 64; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint64_t x = (std::uint64_t{1} << j) | (std::uint64_t{1} << k);
      MATCH(63 - j, LeadingZeroBitCount(x))("j=%d, k=%d", j, k);
    }
  }
  MATCH(32, LeadingZeroBitCount(std::uint32_t{0}));
  for (int j{0}; j < 32; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint32_t x = (std::uint32_t{1} << j) | (std::uint32_t{1} << k);
      MATCH(31 - j, LeadingZeroBitCount(x))("j=%d, k=%d", j, k);
    }
  }
  MATCH(16, LeadingZeroBitCount(std::uint16_t{0}));
  for (int j{0}; j < 16; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint16_t x = (std::uint16_t{1} << j) | (std::uint16_t{1} << k);
      MATCH(15 - j, LeadingZeroBitCount(x))("j=%d, k=%d", j, k);
    }
  }
  MATCH(8, LeadingZeroBitCount(std::uint8_t{0}));
  for (int j{0}; j < 8; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint8_t x = (std::uint8_t{1} << j) | (std::uint8_t{1} << k);
      MATCH(7 - j, LeadingZeroBitCount(x))("j=%d, k=%d", j, k);
    }
  }
  return testing::Complete();
}
