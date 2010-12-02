// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-DEFAULT
// RUN: %clang_cc1 %s -fno-common -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-NOCOMMON

// CHECK-DEFAULT: @x = common global
// CHECK-NOCOMMON: @x = global
int x;
