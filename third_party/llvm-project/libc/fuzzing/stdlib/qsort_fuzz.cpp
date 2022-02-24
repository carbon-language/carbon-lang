//===-- qsort_fuzz.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc qsort implementation.
///
//===----------------------------------------------------------------------===//

#include "src/stdlib/qsort.h"
#include <stdint.h>

static int int_compare(const void *l, const void *r) {
  int li = *reinterpret_cast<const int *>(l);
  int ri = *reinterpret_cast<const int *>(r);
  if (li == ri)
    return 0;
  else if (li > ri)
    return 1;
  else
    return -1;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const size_t array_size = size / sizeof(int);
  if (array_size == 0)
    return 0;

  int *array = new int[array_size];
  const int *data_as_int = reinterpret_cast<const int *>(data);
  for (size_t i = 0; i < array_size; ++i)
    array[i] = data_as_int[i];

  __llvm_libc::qsort(array, array_size, sizeof(int), int_compare);

  for (size_t i = 0; i < array_size - 1; ++i) {
    if (array[i] > array[i + 1])
      __builtin_trap();
  }

  delete[] array;
  return 0;
}
