// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -emit-llvm %s -o - | FileCheck %s

// rdar://problem/10290317
@interface Test0
- (void) setValue: (id) x;
@end
void test0(Test0 *t0, id value) {
  t0.value = value;
}
// CHECK: define void @test0(
// CHECK: call i8* @objc_retain(
// CHECK: call i8* @objc_retain(
// CHECK: @objc_msgSend
// CHECK: call void @objc_release(
// CHECK: call void @objc_release(
