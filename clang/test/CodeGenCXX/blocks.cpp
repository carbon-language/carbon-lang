// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

namespace test0 {
  // CHECK: define void @_ZN5test04testEi(
  // CHECK: define internal void @__test_block_invoke_{{.*}}(
  // CHECK: define internal void @__block_global_{{.*}}(
  void test(int x) {
    ^{ ^{ (void) x; }; };
  }
}

extern void (^out)();

namespace test1 {
  // Capturing const objects doesn't require a local block.
  // CHECK: define void @_ZN5test15test1Ev()
  // CHECK:   store void ()* bitcast ({{.*}} @__block_literal_global{{.*}} to void ()*), void ()** @out
  void test1() {
    const int NumHorsemen = 4;
    out = ^{ (void) NumHorsemen; };
  }

  // That applies to structs too...
  // CHECK: define void @_ZN5test15test2Ev()
  // CHECK:   store void ()* bitcast ({{.*}} @__block_literal_global{{.*}} to void ()*), void ()** @out
  struct loc { double x, y; };
  void test2() {
    const loc target = { 5, 6 };
    out = ^{ (void) target; };
  }

  // ...unless they have mutable fields...
  // CHECK: define void @_ZN5test15test3Ev()
  // CHECK:   [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
  // CHECK:   [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK:   store void ()* [[T0]], void ()** @out
  struct mut { mutable int x; };
  void test3() {
    const mut obj = { 5 };
    out = ^{ (void) obj; };
  }

  // ...or non-trivial destructors...
  // CHECK: define void @_ZN5test15test4Ev()
  // CHECK:   [[OBJ:%.*]] = alloca
  // CHECK:   [[BLOCK:%.*]] = alloca [[BLOCK_T:%.*]],
  // CHECK:   [[T0:%.*]] = bitcast [[BLOCK_T]]* [[BLOCK]] to void ()*
  // CHECK:   store void ()* [[T0]], void ()** @out
  struct scope { int x; ~scope(); };
  void test4() {
    const scope obj = { 5 };
    out = ^{ (void) obj; };
  }

  // ...or non-trivial copy constructors, but it's not clear how to do
  // that and still have a constant initializer in '03.
}
