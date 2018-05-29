#include "../../lib/evaluate/leading-zero-bit-count.h"
#include "testing.h"
#include <cinttypes>
#include <cstdlib>
#include <iostream>

using Fortran::evaluate::LeadingZeroBitCount;

int main() {
  CHECK(64, LeadingZeroBitCount(std::uint64_t{0}));
  for (int j{0}; j < 64; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint64_t x = (std::uint64_t{1} << j) | (std::uint64_t{1} << k);
      CHECK_CASE(x, 63 - j, LeadingZeroBitCount(x));
    }
  }
  CHECK(32, LeadingZeroBitCount(std::uint32_t{0}));
  for (int j{0}; j < 32; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint32_t x = (std::uint32_t{1} << j) | (std::uint32_t{1} << k);
      CHECK_CASE(x, 31 - j, LeadingZeroBitCount(x));
    }
  }
  CHECK(16, LeadingZeroBitCount(std::uint16_t{0}));
  for (int j{0}; j < 16; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint16_t x = (std::uint16_t{1} << j) | (std::uint16_t{1} << k);
      CHECK_CASE(x, 15 - j, LeadingZeroBitCount(x));
    }
  }
  CHECK(8, LeadingZeroBitCount(std::uint8_t{0}));
  for (int j{0}; j < 8; ++j) {
    for (int k{0}; k < j; ++k) {
      std::uint8_t x = (std::uint8_t{1} << j) | (std::uint8_t{1} << k);
      CHECK_CASE(x, 7 - j, LeadingZeroBitCount(x));
    }
  }
  return testing::Complete();
}
