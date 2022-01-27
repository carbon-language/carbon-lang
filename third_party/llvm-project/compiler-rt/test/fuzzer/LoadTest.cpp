// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Simple test for a fuzzer: find interesting value of array index.
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

static volatile uint8_t Sink;
const int kArraySize = 32505854; // 0x01effffe
uint8_t array[kArraySize];

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size != 8)
    return 0;
  uint64_t a = 0;
  memcpy(&a, Data, sizeof(a));
  a &= 0x1fffffff;
  Sink = array[a % (kArraySize + 1)];
  return 0;
}
