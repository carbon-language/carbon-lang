// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@protocol FooTestProtocol
+ protocolClassMethod;
- protocolInstanceMethod;
@end
@interface Foo <FooTestProtocol> {
  void *isa;
}
+ (int)classMethod1:a withKeyword:b;
+ (void)classMethod2;
+ new;
- instanceMethod1;
@end

@interface Foo (FooTestCategory)
+ categoryClassMethod;
- categoryInstanceMethod;
@end

void func() {
  Foo *obj = [Foo new];
  [obj xx];
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:23:19 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: categoryClassMethod : 0
// CHECK-CC1: classMethod1:withKeyword: : 0
// CHECK-CC1: classMethod2 : 0
// CHECK-CC1: new : 0
// CHECK-CC1: protocolClassMethod : 0
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:24:8 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: categoryInstanceMethod : 0
// CHECK-CC2: instanceMethod1 : 0
// CHECK-CC2: protocolInstanceMethod : 0
