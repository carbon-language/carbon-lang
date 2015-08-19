// RUN: %clang_cc1 -triple i686-pc-windows-gnu -S %s  -o - | FileCheck %s -check-prefix=CHECK_I686
// CHECK_I686: lda,12
// RUN: %clang_cc1 -triple x86_64-pc-windows-gnu -S %s  -o - | FileCheck %s -check-prefix=CHECK_X86_64
// CHECK_X86_64: lda,16
long double lda;
