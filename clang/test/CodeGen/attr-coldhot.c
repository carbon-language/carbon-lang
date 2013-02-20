// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

int test1() __attribute__((__cold__)) {
  return 42;

// Check that we set the optsize attribute on the function.
// CHECK: @test1{{.*}}#0
// CHECK: ret
}

// CHECK: attributes #0 = { nounwind optsize "target-features"={{.*}} }
