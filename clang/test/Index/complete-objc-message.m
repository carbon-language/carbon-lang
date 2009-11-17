// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@protocol FooTestProtocol
+ protocolClassMethod;
- protocolInstanceMethod : (int)value;
@end
@interface Foo <FooTestProtocol> {
  void *isa;
}
+ (int)classMethod1:a withKeyword:(int)b;
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
// RUN: c-index-test -code-completion-at=%s:23:19 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText categoryClassMethod}
// CHECK-CC1: {TypedText classMethod2}
// CHECK-CC1: {TypedText new}
// CHECK-CC1: {TypedText protocolClassMethod}
// CHECK-CC1: {TypedText classMethod1:}{Placeholder (id)a}{Text  withKeyword:}{Placeholder (int)b}
// RUN: c-index-test -code-completion-at=%s:24:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText categoryInstanceMethod}
// CHECK-CC2: {TypedText instanceMethod1}
// CHECK-CC2: {TypedText protocolInstanceMethod:}{Placeholder (int)value}
