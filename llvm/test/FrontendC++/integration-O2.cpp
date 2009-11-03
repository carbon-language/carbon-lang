// RUN: %llvmgxx %s -O2 -S -o - | FileCheck %s
// XFAIL: *

// This test verifies that we get expected codegen out of the -O2 optimization
// level from the full optimizer.



// Verify that ipsccp is running and can eliminate globals.
static int test1g = 42;
void test1f1() {
  if (test1g == 0) test1g = 0;
}
int test1f2() {
  return test1g;
}

// CHECK: @_Z7test1f2v()
// CHECK: entry:
// CHECK-NEXT: ret i32 42
