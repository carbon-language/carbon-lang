// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

namespace test0 {
  // CHECK: define void @_ZN5test04testEi(
  // CHECK: define internal void @__test_block_invoke_{{.*}}(
  // CHECK: define internal void @__block_global_{{.*}}(
  void test(int x) {
    ^{ ^{ (void) x; }; };
  }
}
