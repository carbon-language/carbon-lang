// RUN: %clang_cc1 -fexceptions -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck -check-prefix NOEXC %s

int opaque();

// CHECK:       define [[INT:i.*]] @test0() {
// CHECK-NOEXC: define [[INT:i.*]] @test0() nounwind {
int test0(void) {
  return opaque();
}

// <rdar://problem/8087431>: locally infer nounwind at -O0
// CHECK:       define [[INT:i.*]] @test1() nounwind {
// CHECK-NOEXC: define [[INT:i.*]] @test1() nounwind {
int test1(void) {
}
