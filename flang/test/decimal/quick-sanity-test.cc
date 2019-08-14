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

void testDirect(float x, const char *expect, int expectExpo, int flags = 0) {
  char buffer[1024];
  ++tests;
  auto result{ConvertFloatToDecimal(buffer, sizeof buffer,
      static_cast<enum DecimalConversionFlags>(flags), 1024, RoundNearest, x)};
  if (result.str == nullptr) {
    failed(x) << ' ' << flags << ": no result str\n";
  } else if (std::strcmp(result.str, expect) != 0 ||
      result.decimalExponent != expectExpo) {
    failed(x) << ' ' << flags << ": expect '." << expect << 'e' << expectExpo
              << "', got '." << result.str << 'e' << result.decimalExponent
              << "'\n";
  }
}

void testReadback(float x, int flags) {
  char buffer[1024];
  ++tests;
  auto result{ConvertFloatToDecimal(buffer, sizeof buffer,
      static_cast<enum DecimalConversionFlags>(flags), 1024, RoundNearest, x)};
  if (result.str == nullptr) {
    failed(x) << ' ' << flags << ": no result str\n";
  } else {
    float y{0};
    char *q{const_cast<char *>(result.str)};
    int expo{result.decimalExponent};
    expo -= result.length;
    if (*q == '-' || *q == '+') {
      ++expo;
    }
    std::sprintf(q + result.length, "e%d", expo);
    const char *p{q};
    auto flags{ConvertDecimalToFloat(&p, &y, RoundNearest)};
    if (x != y || *p != '\0' || (flags & Invalid)) {
      failed(x) << ' ' << flags << ": -> '" << buffer << "' -> 0x" << std::hex
                << *reinterpret_cast<std::uint32_t *>(&y) << std::dec << " '"
                << p << "'\n";
    }
  }
}

int main() {
  float x;
  std::uint32_t *ix{reinterpret_cast<std::uint32_t *>(&x)};
  testDirect(-1.0, "-1", 1);
  testDirect(0.0, "0", 0);
  testDirect(0.0, "+0", 0, AlwaysSign);
  testDirect(1.0, "1", 1);
  testDirect(2.0, "2", 1);
  testDirect(-1.0, "-1", 1);
  testDirect(314159, "314159", 6);
  testDirect(0.0625, "625", -1);
  *ix = 0x80000000;
  testDirect(x, "-0", 0);
  *ix = 0x7f800000;
  testDirect(x, "Inf", 0);
  testDirect(x, "+Inf", 0, AlwaysSign);
  *ix = 0xff800000;
  testDirect(x, "-Inf", 0);
  *ix = 0xffffffff;
  testDirect(x, "NaN", 0);
  testDirect(x, "NaN", 0, AlwaysSign);
  *ix = 1;
  testDirect(x,
      "140129846432481707092372958328991613128026194187651577175706828388979108"
      "268586060148663818836212158203125",
      -44, 0);
  testDirect(x, "1", -44, Minimize);
  *ix = 0x7f777777;
  testDirect(x, "3289396118917826996438159226753253376", 39, 0);
  testDirect(x, "32893961", 39, Minimize);
  for (*ix = 0; *ix < 16; ++*ix) {
    testReadback(x, 0);
    testReadback(-x, 0);
    testReadback(x, Minimize);
    testReadback(-x, Minimize);
  }
  for (*ix = 1; *ix < 0x7f800000; *ix *= 2) {
    testReadback(x, 0);
    testReadback(-x, 0);
    testReadback(x, Minimize);
    testReadback(-x, Minimize);
  }
  for (*ix = 0x7f7ffff0; *ix < 0x7f800000; ++*ix) {
    testReadback(x, 0);
    testReadback(-x, 0);
    testReadback(x, Minimize);
    testReadback(-x, Minimize);
  }
  for (*ix = 0; *ix < 0x7f800000; *ix += 65536) {
    testReadback(x, 0);
    testReadback(-x, 0);
    testReadback(x, Minimize);
    testReadback(-x, Minimize);
  }
  for (*ix = 0; *ix < 0x7f800000; *ix += 99999) {
    testReadback(x, 0);
    testReadback(-x, 0);
    testReadback(x, Minimize);
    testReadback(-x, Minimize);
  }
  for (*ix = 0; *ix < 0x7f800000; *ix += 32767) {
    testReadback(x, 0);
    testReadback(-x, 0);
    testReadback(x, Minimize);
    testReadback(-x, Minimize);
  }
  std::cout << tests << " tests run, " << fails << " tests failed\n";
  return fails > 0;
}
