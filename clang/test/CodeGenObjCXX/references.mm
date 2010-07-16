// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct A { ~A(); };

@interface B {
  A a;
}

- (const A&)getA;
@end

@implementation B 

- (const A&)getA {
  return a;
}

@end

// CHECK: define void @_Z1fP1B
// CHECK: objc_msgSend to
// CHECK-NOT: call void @_ZN1AD1Ev
// CHECK: ret void
void f(B* b) {
  (void)[b getA];
}
