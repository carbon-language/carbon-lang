// Test if PGO sample use passes are invoked.
//
// Ensure Pass SampleProfileLoader is invoked.
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -mllvm -debug-pass=Structure -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=STRUCTURE
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -mllvm -debug-pass=Structure -mllvm -inline-threshold=0 -emit-llvm -o - 2>&1 | FileCheck %s
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample.prof %s -mllvm -debug-pass=Structure -mllvm -inline-threshold=0 -emit-llvm -fexperimental-new-pass-manager -o - 2>&1 | FileCheck %s
// STRUCTURE: Remove unused exception handling info
// STRUCTURE: Sample profile pass

void baz();

// CHECK-LABEL: @callee(
void callee(int t) {
  for (int i = 0; i < t; i++)
    baz();
}

// CHECK-LABEL: @bar(
// cold call to callee should not be inlined.
// CHECK: call void @callee
void bar(int x) {
  if (x < 100)
    callee(x);
}

// CHECK-LABEL: @foo(
// bar should be early-inlined because it is hot inline instance in profile.
// callee should be inlined because it is hot callsite in the inline instance
// of foo:bar.
// CHECK-NOT: call void @callee
// CHECK-NOT: call void @bar
void foo(int x) {
  bar(x);
}
