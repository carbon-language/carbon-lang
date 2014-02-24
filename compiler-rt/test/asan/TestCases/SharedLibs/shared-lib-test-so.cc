//===----------- shared-lib-test-so.cc --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>
#include <string.h>

int pad[10];
int GLOB[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

extern "C"
void inc(int index) {
  GLOB[index]++;
}

extern "C"
void inc2(int *a, int index) {
  a[index]++;
}

extern "C"
void my_memset(void *p, size_t sz) {
  memset(p, 0, sz);
}
