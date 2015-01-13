// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -gline-tables-only -fblocks -emit-llvm %s -o - | FileCheck %s

void fn();

struct foo {
  ~foo();
};

void func() {
  ^{
    foo f;
    fn();
    // CHECK: cleanup, !dbg [[LINE:![0-9]*]]
    // CHECK: [[LINE]] = !{i32 [[@LINE+1]], 
  }();
}
