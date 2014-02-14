//===----------- dlclose-test-so.cc -----------------------------*- C++ -*-===//
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
// Regression test for
// http://code.google.com/p/address-sanitizer/issues/detail?id=19
//===----------------------------------------------------------------------===//
#include <stdio.h>

static int pad1;
static int static_var;
static int pad2;

extern "C"
int *get_address_of_static_var() {
  return &static_var;
}

__attribute__((constructor))
void at_dlopen() {
  printf("%s: I am being dlopened\n", __FILE__);
}
__attribute__((destructor))
void at_dlclose() {
  printf("%s: I am being dlclosed\n", __FILE__);
}
