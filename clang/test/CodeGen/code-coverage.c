// RUN: %clang_cc1 -O0  -mno-red-zone -fprofile-arcs -ftest-coverage -emit-llvm %s -o - | FileCheck %s
// <rdar://problem/12843084>

int test1(int a) {
  switch (a % 2) {
  case 0:
    ++a;
  case 1:
    a /= 2;
  }
  return a;
}

// Check tha the `-mno-red-zone' flag is set here on the generated functions.

// CHECK: void @__llvm_gcov_indirect_counter_increment(i32* %{{.*}}, i64** %{{.*}}) unnamed_addr noinline noredzone
// CHECK: void @__llvm_gcov_writeout() unnamed_addr noinline noredzone
// CHECK: void @__llvm_gcov_init() unnamed_addr noinline noredzone
// CHECK: void @__gcov_flush() unnamed_addr noinline noredzone
