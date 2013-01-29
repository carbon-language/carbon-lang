//===-- msandr_test_so.cc  ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
// MemorySanitizer unit tests.
//===----------------------------------------------------------------------===//

#include "msandr_test_so.h"

void dso_memfill(char* s, unsigned n) {
  for (unsigned i = 0; i < n; ++i)
    s[i] = i;
}

int dso_callfn(int (*fn)(void)) {
  volatile int x = fn();
  return x;
}

int dso_callfn1(int (*fn)(long long, long long, long long)) {  //NOLINT
  volatile int x = fn(1, 2, 3);
  return x;
}

int dso_stack_store(void (*fn)(int*, int*), int x) {
  int y = x + 1;
  fn(&x, &y);
  return y;
}

void break_optimization(void *x) {}
