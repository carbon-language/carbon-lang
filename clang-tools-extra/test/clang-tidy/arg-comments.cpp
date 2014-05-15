// RUN: clang-tidy --checks='-*,misc-argument-comment' %s -- | FileCheck %s

// FIXME: clang-tidy should provide a -verify mode to make writing these checks
// easier and more accurate.

// CHECK-NOT: warning

void f(int x, int y);

void ffff(int xxxx, int yyyy);

void g() {
  // CHECK: [[@LINE+5]]:5: warning: argument name 'y' in comment does not match parameter name 'x'
  // CHECK: :8:12: note: 'x' declared here
  // CHECK: [[@LINE+3]]:14: warning: argument name 'z' in comment does not match parameter name 'y'
  // CHECK: :8:19: note: 'y' declared here
  // CHECK-NOT: warning
  f(/*y=*/0, /*z=*/0);
}

// CHECK-NOT: warning
