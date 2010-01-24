// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

const int x = 10;
const int y = 20;
// CHECK-NOT: @x
// CHECK: @_ZL1y = internal constant i32 20
const int& b() { return y; }

const char z1[] = "asdf";
const char z2[] = "zxcv";
// CHECK-NOT: @z1
// CHECK: @_ZL2z2 = internal constant
const char* b2() { return z2; }
