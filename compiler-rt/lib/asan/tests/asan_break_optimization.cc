//===-- asan_break_optimization.cc ----------------------===//
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

#include "asan_test_utils.h"
// Have this function in a separate file to avoid inlining.
// (Yes, we know about cross-file inlining, but let's assume we don't use it).
extern "C" void break_optimization(void *x) {
}
