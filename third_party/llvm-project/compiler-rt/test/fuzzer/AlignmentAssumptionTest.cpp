// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test for alignment assumption failure.

#include <assert.h>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

static volatile int32_t Sink;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(Data);
  if (Size > 0 && Data[0] == 'H') {
    Sink = 1;
    if (Size > 1 && Data[1] == 'i') {
      Sink = 2;
      if (Size > 2 && Data[2] == '!') {
        auto r = __builtin_assume_aligned(Data + 1, 0x8000);
      }
    }
  }
  return 0;
}
