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

@interface MyClass { }
+ (int)MyClassMethod:(id)obj;
- (int)MyInstMethod:(id)x second:(id)y;
@end

@interface MySubClass : MyClass { }
+ (int)MySubClassMethod;
- (int)MySubInstMethod;
@end

@implementation MyClass 
+ (int)MyClassMethod:(id)obj {
  return 0;
}

+ (int)MyPrivateMethod {
  return 1;
}

- (int)MyInstMethod:(id)x second:(id)y {
  return 2;
}

- (int)MyPrivateInstMethod {
  return 3;
}
@end

@implementation MySubClass
+ (int)MySubClassMethod {
  return 2;
}

+ (int)MySubPrivateMethod {
  return [super MyPrivateMethod];
}

- (int)MySubInstMethod:(id)obj {
  return [super MyInstMethod: obj second:obj];
}

- (int)MyInstMethod:(id)x second:(id)y {
  return 3;
}
@end

void test_super_var(MySubClass *super) {
  [super MyInstMethod: super second:super];
}

@protocol FooTestProtocol2
- (int)secondProtocolInstanceMethod;
@end

void test_qual_id(id<FooTestProtocol,FooTestProtocol2> ptr) {
  [ptr protocolInstanceMethod:1];
}

// RUN: c-index-test -code-completion-at=%s:23:19 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText categoryClassMethod}
// CHECK-CC1: {TypedText classMethod1:}{Placeholder (id)a}{Text  withKeyword:}{Placeholder (int)b}
// CHECK-CC1: {TypedText classMethod2}
// CHECK-CC1: {TypedText new}
// CHECK-CC1: {TypedText protocolClassMethod}
// RUN: c-index-test -code-completion-at=%s:24:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText categoryInstanceMethod}
// CHECK-CC2: {TypedText instanceMethod1}
// CHECK-CC2: {TypedText protocolInstanceMethod:}{Placeholder (int)value}
// RUN: c-index-test -code-completion-at=%s:61:16 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCClassMethodDecl:{TypedText MyClassMethod:}{Placeholder (id)obj}
// CHECK-CC3: ObjCClassMethodDecl:{TypedText MyPrivateMethod}
// RUN: c-index-test -code-completion-at=%s:65:16 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInstanceMethodDecl:{TypedText MyInstMethod:}{Placeholder (id)x}{Text  second:}{Placeholder (id)y}
// CHECK-CC4: ObjCInstanceMethodDecl:{TypedText MyPrivateInstMethod}
// RUN: c-index-test -code-completion-at=%s:74:9 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText MyInstMethod:}{Placeholder (id)x}{Text  second:}{Placeholder (id)y}
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText MySubInstMethod}
// RUN: c-index-test -code-completion-at=%s:82:8 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText protocolInstanceMethod:}{Placeholder (int)value}
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText secondProtocolInstanceMethod}

