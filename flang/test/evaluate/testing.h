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

#ifndef FORTRAN_EVALUATE_TESTING_H_
#define FORTRAN_EVALUATE_TESTING_H_

#include <cinttypes>
#include <cstdlib>
#include <iostream>

namespace testing {

int passes{0};
int failures{0};

void Check(const char *file, int line, std::uint64_t want, std::uint64_t got) {
  if (want != got) {
    ++failures;
    std::cerr << file << ':' << std::dec << line << '(' << (passes + failures)
              << "): want 0x" << std::hex << want
              << ", got 0x" << got << '\n' << std::dec;
  } else {
    ++passes;
  }
}

void Check(const char *file, int line, std::uint64_t x, std::uint64_t want,
           std::uint64_t got) {
  if (want != got) {
    ++failures;
    std::cerr << file << ':' << std::dec << line << '(' << (passes + failures)
              << ")[0x" << std::hex << x << "]: want 0x" << want
              << ", got 0x" << got << '\n' << std::hex;
  } else {
    ++passes;
  }
}

int Complete() {
  if (failures == 0) {
    if (passes == 1) {
      std::cout << "test PASSES\n";
    } else {
      std::cout << "all " << std::dec << passes << " tests PASS\n";
    }
    passes = 0;
    return EXIT_SUCCESS;
  } else {
    if (passes == 1) {
      std::cerr << std::dec << "1 test passes, ";
    } else {
      std::cerr << std::dec << passes << " tests pass, ";
    }
    if (failures == 1) {
      std::cerr << "1 test FAILS\n";
    } else {
      std::cerr << std::dec << failures << " tests FAIL\n";
    }
    passes = failures = 0;
    return EXIT_FAILURE;
  }
}

}  // namespace testing

#define CHECK(want, got) \
  testing::Check(__FILE__, __LINE__, (want), (got))
#define CHECK_CASE(n, want, got) \
  testing::Check(__FILE__, __LINE__, (n), (want), (got))

#endif  // FORTRAN_EVALUATE_TESTING_H_
