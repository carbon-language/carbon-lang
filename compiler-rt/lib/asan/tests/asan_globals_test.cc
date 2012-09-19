//===-- asan_globals_test.cc ----------------------------------------------===//
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
// Some globals in a separate file.
//===----------------------------------------------------------------------===//

extern char glob5[5];
static char static10[10];

int GlobalsTest(int zero) {
  static char func_static15[15];
  glob5[zero] = 0;
  static10[zero] = 0;
  func_static15[zero] = 0;
  return glob5[1] + func_static15[2];
}
