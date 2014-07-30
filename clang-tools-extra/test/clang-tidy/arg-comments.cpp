// RUN: clang-tidy --checks='-*,misc-argument-comment' %s -- -std=c++11 | FileCheck %s -implicit-check-not='{{warning:|error:}}'

// FIXME: clang-tidy should provide a -verify mode to make writing these checks
// easier and more accurate.

void ffff(int xxxx, int yyyy);

void f(int x, int y);
void g() {
  // CHECK: [[@LINE+5]]:5: warning: argument name 'y' in comment does not match parameter name 'x'
  // CHECK: :[[@LINE-3]]:12: note: 'x' declared here
  // CHECK: [[@LINE+3]]:14: warning: argument name 'z' in comment does not match parameter name 'y'
  // CHECK: :[[@LINE-5]]:19: note: 'y' declared here
  // CHECK-NOT: warning
  f(/*y=*/0, /*z=*/0);
}

struct Closure {};

template <typename T1, typename T2>
Closure *NewCallback(void (*f)(T1, T2), T1 arg1, T2 arg2) { return nullptr; }

template <typename T1, typename T2>
Closure *NewPermanentCallback(void (*f)(T1, T2), T1 arg1, T2 arg2) { return nullptr; }

void h() {
  (void)NewCallback(&ffff, /*xxxx=*/11, /*yyyy=*/22);
  (void)NewPermanentCallback(&ffff, /*xxxx=*/11, /*yyyy=*/22);
}
