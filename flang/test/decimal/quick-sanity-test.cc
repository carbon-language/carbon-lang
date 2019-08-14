// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

#include "../../lib/decimal/decimal.h"
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <iostream>

static int tests{0};
static int fails{0};

std::ostream &failed(float x) {
  ++fails;
  return std::cout << "FAIL: 0x" << std::hex
      << *reinterpret_cast<std::uint32_t *>(&x) << std::dec;
}

void testDirect(float x, const char *expect, int expectExpo) {
  char buffer[1024];
  ++tests;
  auto result{ConvertFloatToDecimal(buffer, sizeof buffer,
      static_cast<enum DecimalConversionFlags>(0), 1024,
      RoundNearest, x)};
  if (result.str == nullptr) {
    failed(x) << ": no result str\n";
  } else if (std::strcmp(result.str, expect) != 0) {
    failed(x) << ": expect '" << expect << "', got '" << result.str << "'\n";
  } else if (result.decimalExponent != expectExpo) {
    failed(x) << ": expect exponent " << expectExpo << ", got "
        << result.decimalExponent << '\n';
  }
}

void testReadback(float x) {
  char buffer[1024];
  ++tests;
  auto result{ConvertFloatToDecimal(buffer, sizeof buffer,
      static_cast<enum DecimalConversionFlags>(0), 1024,
      RoundNearest, x)};
  if (result.str == nullptr) {
    failed(x) << ": no result str\n";
  } else {
    float y{0};
    char *q{const_cast<char *>(result.str)};
    std::sprintf(q + result.length, "e%d", static_cast<int>(result.decimalExponent - result.length));
    const char *p{q};
    auto flags{ConvertDecimalToFloat(&p, &y, RoundNearest)};
    if (x != y || *p != '\0' || (flags & Invalid)) {
      failed(x) << ": -> '" << buffer << "' -> 0x"
          << std::hex << *reinterpret_cast<std::uint32_t *>(&y)
          << std::dec << " '" << p << "'\n";
    }
  }
}

int main() {
  testDirect(-1.0, "-1", 1);
  testDirect(0.0, "0", 0);
  testDirect(1.0, "1", 1);
  testDirect(2.0, "2", 1);
  testDirect(-1.0, "-1", 1);
  testDirect(314159, "314159", 6);
  testDirect(0.0625, "625", -1);
  float x;
  std::uint32_t *ix{reinterpret_cast<std::uint32_t *>(&x)};
  *ix = 0x80000000;
  testDirect(x, "-0", 0);
  *ix = 0x7f800000;
  testDirect(x, "Inf", 0);
  *ix = 0xff800000;
  testDirect(x, "-Inf", 0);
  *ix = 0xffffffff;
  testDirect(x, "NaN", 0);
  std::cout << tests << " tests run, " << fails << " tests failed\n";
  return fails > 0;
}
