// RUN: clang-cc -emit-llvm -o - %s | FileCheck %s

const int x = 10;
const int y = 20;
// CHECK-NOT: @x
// CHECK: @y = internal constant i32 20
const int& b() { return y; }

