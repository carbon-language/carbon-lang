// RUN: %clang_cc1 -emit-llvm -disable-red-zone -femit-coverage-data %s -o - | FileCheck %s

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

// Check that the noredzone flag is set on the generated functions.

// CHECK: void @__llvm_gcov_indirect_counter_increment(i32* %{{.*}}, i64** %{{.*}}) unnamed_addr #1
// CHECK: void @__llvm_gcov_writeout() unnamed_addr #1
// CHECK: void @__llvm_gcov_init() unnamed_addr #1
// CHECK: void @__gcov_flush() unnamed_addr #1

// CHECK: attributes #0 = { noredzone nounwind "target-features"={{.*}} }
// CHECK: attributes #1 = { noinline noredzone }
