// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macosx10.7.0 -emit-llvm -o - %s | FileCheck %s

// CHECK: define linkonce_odr void @_Z11inline_funci
inline void inline_func(int n) {
  // CHECK: call i32 @_ZZ11inline_funciENKUlvE_clEv
  int i = []{ return 1; }();

  // CHECK: call i32 @_ZZ11inline_funciENKUlvE0_clEv
  int j = [=] { return n + i; }();

  // CHECK: call double @_ZZ11inline_funciENKUlvE1_clEv
  int k = [=] () -> double { return n + i; }();

  // CHECK: call i32 @_ZZ11inline_funciENKUliE_clEi
  int l = [=] (int x) -> int { return x + i; }(n);

  // CHECK: ret void
}

void call_inline_func() {
  inline_func(17);
}
