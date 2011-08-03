// RUN: %clang_cc1 -triple s390x-unknown-linux -emit-llvm -o - %s | FileCheck %s

// Test that we don't crash.  The s390x-unknown-linux target happens
// to need to set a sext argument attribute on this call, and we need
// to make sure that rewriting it correctly drops that attribute when
// also dropping the spurious argument.
void test0_helper();
void test0() {
  // CHECK: call void @test0_helper()
  test0_helper(1);
}
void test0_helper() {}

