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

#include "testing.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>

namespace testing {

namespace {
int passes{0};
int failures{0};
}  // namespace

static void BitBucket(const char *, ...) {}

static void PrintFailureDetails(const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  fputs("\t", stderr);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fputc('\n', stderr);
}

FailureDetailPrinter Test(
    const char *file, int line, const char *predicate, bool pass) {
  if (pass) {
    ++passes;
    return BitBucket;
  } else {
    ++failures;
    fprintf(stderr, "%s:%d: FAIL %s\n", file, line, predicate);
    return PrintFailureDetails;
  }
}

FailureDetailPrinter Compare(const char *file, int line, const char *xs,
    const char *rel, const char *ys, unsigned long long x,
    unsigned long long y) {
  while (*rel == ' ') {
    ++rel;
  }
  bool pass{false};
  if (*rel == '<') {
    if (rel[1] == '=') {
      pass = x <= y;
    } else {
      pass = x < y;
    }
  } else if (*rel == '>') {
    if (rel[1] == '=') {
      pass = x >= y;
    } else {
      pass = x > y;
    }
  } else if (*rel == '=') {
    pass = x == y;
  } else if (*rel == '!') {
    pass = x != y;
  }
  if (pass) {
    ++passes;
    return BitBucket;
  } else {
    ++failures;
    fprintf(stderr, "%s:%d: FAIL %s[0x%llx] %s %s[0x%llx]:\n", file, line, xs,
        x, rel, ys, y);
    return PrintFailureDetails;
  }
}

int Complete() {
  if (failures == 0) {
    if (passes == 1) {
      std::cout << "single test PASSES\n";
    } else {
      std::cout << "all " << std::dec << passes << " tests PASS\n";
    }
    passes = 0;
    return EXIT_SUCCESS;
  } else {
    if (passes == 1) {
      std::cerr << "1 test passes, ";
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
