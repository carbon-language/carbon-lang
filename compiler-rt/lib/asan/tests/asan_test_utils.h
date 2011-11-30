//===-- asan_test_utils.h ------------*- C++ -*-===//
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

#ifndef ASAN_TEST_UTILS_H
#define ASAN_TEST_UTILS_H

// Make the compiler think that something is going on there.
extern "C" void break_optimization(void *);

// This function returns its parameter but in such a way that compiler
// can not prove it.
template<class T>
__attribute__((noinline))
static T Ident(T t) {
  T ret = t;
  break_optimization(&ret);
  return ret;
}

#endif  // ASAN_TEST_UTILS_H
