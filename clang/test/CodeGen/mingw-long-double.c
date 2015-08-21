// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple i686-pc-windows-gnu -S %s  -o - | FileCheck %s -check-prefix=CHECK_I686
// CHECK_I686: _lda,12
// CHECK_I686: _lds,16
// RUN: %clang_cc1 -triple x86_64-pc-windows-gnu -S %s  -o - | FileCheck %s -check-prefix=CHECK_X86_64
// CHECK_X86_64: lda,16
// CHECK_X86_64: lds,32
long double lda;
struct {
  char c;
  long double ldb;
} lds;
