// Note: the run lines follow their respective tests, since line/column
// matter in this test.
#define nil (void*)0
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
MyClass *getMyClass();
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

@interface Overload
- (int)Method:(int)i;
- (int)Method;
- (int)Method:(float)f Arg1:(int)i1 Arg2:(int)i2;
- (int)Method:(float)f Arg1:(int)i1 OtherArg:(id)obj;
- (int)Method:(float)f SomeArg:(int)i1 OtherArg:(id)obj;
- (int)OtherMethod:(float)f Arg1:(int)i1 Arg2:(int)i2;
@end

void test_overload(Overload *ovl) {
  [ovl Method:1 Arg1:1 OtherArg:ovl];
}

@interface Ellipsis
- (int)Method:(int)i, ...; 
- (int)SentinelMethod:(int)i, ... __attribute__((sentinel(0,1)));
@end
void f(Ellipsis *e) {
  [e Method:1, 2, 3];
}

@interface Overload2
+ (int)Method:(int)i;
+ (int)Method;
+ (int)Method:(float)f Arg1:(int)i1 Arg2:(int)i2;
+ (int)Method:(float)f Arg1:(int)i1 OtherArg:(id)obj;
+ (int)Method:(float)f SomeArg:(int)i1 OtherArg:(id)obj;
+ (int)OtherMethod:(float)f Arg1:(int)i1 Arg2:(int)i2;
@end

void test_overload2(void) {
  [Overload2 Method:1 Arg1:1 OtherArg:ovl];
}

void msg_id(id x) {
  [x Method:1 Arg1:1 OtherArg:ovl];
  [[x blarg] Method:1 Arg1:1 OtherArg:ovl];
  [id Method:1 Arg1:1 OtherArg:ovl];
}

@interface A
- (void)method1;
@end 

@interface B : A
- (void)method2;
@end

void test_ranking(B *b) {
  [b method1];
}

// RUN: c-index-test -code-completion-at=%s:23:19 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText categoryClassMethod}
// CHECK-CC1: {TypedText classMethod1:}{Placeholder (id)a}{HorizontalSpace  }{Text withKeyword:}{Placeholder (int)b}
// CHECK-CC1: {TypedText classMethod2}
// CHECK-CC1: {TypedText new}
// CHECK-CC1: {TypedText protocolClassMethod}
// RUN: c-index-test -code-completion-at=%s:24:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText categoryInstanceMethod}
// CHECK-CC2: {TypedText instanceMethod1}
// CHECK-CC2: {TypedText protocolInstanceMethod:}{Placeholder (int)value}
// RUN: c-index-test -code-completion-at=%s:61:16 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCClassMethodDecl:{ResultType int}{TypedText MyClassMethod:}{Placeholder (id)obj}
// CHECK-CC3: ObjCClassMethodDecl:{ResultType int}{TypedText MyPrivateMethod}
// RUN: c-index-test -code-completion-at=%s:65:16 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType int}{TypedText MyInstMethod:}{Placeholder (id)x}{HorizontalSpace  }{Text second:}{Placeholder (id)y}
// CHECK-CC4: ObjCInstanceMethodDecl:{ResultType int}{TypedText MyPrivateInstMethod}
// RUN: c-index-test -code-completion-at=%s:74:9 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCInstanceMethodDecl:{ResultType int}{TypedText MyInstMethod:}{Placeholder (id)x}{HorizontalSpace  }{Text second:}{Placeholder (id)y}
// CHECK-CC5: ObjCInstanceMethodDecl:{ResultType int}{TypedText MySubInstMethod}
// RUN: c-index-test -code-completion-at=%s:82:8 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInstanceMethodDecl:{ResultType id}{TypedText protocolInstanceMethod:}{Placeholder (int)value}
// CHECK-CC6: ObjCInstanceMethodDecl:{ResultType int}{TypedText secondProtocolInstanceMethod}
// RUN: c-index-test -code-completion-at=%s:95:8 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method}
// CHECK-CC7: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (int)i}
// CHECK-CC7: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// CHECK-CC7: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CC7: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text SomeArg:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CC7: ObjCInstanceMethodDecl:{ResultType int}{TypedText OtherMethod:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// RUN: c-index-test -code-completion-at=%s:95:17 %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType int}{Informative Method:}{TypedText }
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType int}{Informative Method:}{TypedText Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType int}{Informative Method:}{TypedText Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType int}{Informative Method:}{TypedText SomeArg:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// RUN: c-index-test -code-completion-at=%s:95:24 %s | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9: ObjCInstanceMethodDecl:{ResultType int}{Informative Method:}{Informative Arg1:}{TypedText Arg2:}{Placeholder (int)i2}
// CHECK-CC9: ObjCInstanceMethodDecl:{ResultType int}{Informative Method:}{Informative Arg1:}{TypedText OtherArg:}{Placeholder (id)obj}
// RUN: c-index-test -code-completion-at=%s:61:11 %s | FileCheck -check-prefix=CHECK-CCA %s
// CHECK-CCA: TypedefDecl:{TypedText Class}
// CHECK-CCA-NEXT: ObjCInterfaceDecl:{TypedText Foo}
// CHECK-CCA-NOT: FunctionDecl:{ResultType void}{TypedText func}{LeftParen (}{RightParen )}
// CHECK-CCA:FunctionDecl:{ResultType MyClass *}{TypedText getMyClass}{LeftParen (}{RightParen )}
// CHECK-CCA: TypedefDecl:{TypedText id}
// CHECK-CCA: ObjCInterfaceDecl:{TypedText MyClass}
// CHECK-CCA: ObjCInterfaceDecl:{TypedText MySubClass}
// CHECK-CCA: {ResultType Class}{TypedText self}
// CHECK-CCA: {TypedText super}
// RUN: c-index-test -code-completion-at=%s:103:6 %s | FileCheck -check-prefix=CHECK-CCB %s
// CHECK-CCB: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (int)i}{Placeholder , ...}
// CHECK-CCB: ObjCInstanceMethodDecl:{ResultType int}{TypedText SentinelMethod:}{Placeholder (int)i}{Placeholder , ...}{Text , nil}
// RUN: c-index-test -code-completion-at=%s:116:14 %s | FileCheck -check-prefix=CHECK-CCC %s
// CHECK-CCC: ObjCClassMethodDecl:{ResultType int}{TypedText Method}
// CHECK-CCC: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (int)i}
// CHECK-CCC: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// CHECK-CCC: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CCC: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text SomeArg:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CCC: ObjCClassMethodDecl:{ResultType int}{TypedText OtherMethod:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// RUN: c-index-test -code-completion-at=%s:116:23 %s | FileCheck -check-prefix=CHECK-CCD %s
// CHECK-CCD: ObjCClassMethodDecl:{ResultType int}{Informative Method:}{TypedText }
// CHECK-CCD: ObjCClassMethodDecl:{ResultType int}{Informative Method:}{TypedText Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// CHECK-CCD: ObjCClassMethodDecl:{ResultType int}{Informative Method:}{TypedText Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CCD: ObjCClassMethodDecl:{ResultType int}{Informative Method:}{TypedText SomeArg:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// RUN: c-index-test -code-completion-at=%s:116:30 %s | FileCheck -check-prefix=CHECK-CCE %s
// CHECK-CCE: ObjCClassMethodDecl:{ResultType int}{Informative Method:}{Informative Arg1:}{TypedText Arg2:}{Placeholder (int)i2}
// CHECK-CCE: ObjCClassMethodDecl:{ResultType int}{Informative Method:}{Informative Arg1:}{TypedText OtherArg:}{Placeholder (id)obj}
// RUN: c-index-test -code-completion-at=%s:61:11 %s | FileCheck -check-prefix=CHECK-CCF %s
// CHECK-CCF: TypedefDecl:{TypedText Class}
// CHECK-CCF: ObjCInterfaceDecl:{TypedText Foo}
// CHECK-CCF-NOT: FunctionDecl:{ResultType void}{TypedText func}{LeftParen (}{RightParen )}
// CHECK-CCF: TypedefDecl:{TypedText id}
// CHECK-CCF: ObjCInterfaceDecl:{TypedText MyClass}
// CHECK-CCF: ObjCInterfaceDecl:{TypedText MySubClass}
// CHECK-CCF: {ResultType Class}{TypedText self}
// CHECK-CCF: {TypedText super}
// RUN: c-index-test -code-completion-at=%s:120:6 %s | FileCheck -check-prefix=CHECK-CCG %s
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType id}{TypedText categoryInstanceMethod}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType id}{TypedText instanceMethod1}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (int)i}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text SomeArg:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText MyInstMethod:}{Placeholder (id)x}{HorizontalSpace  }{Text second:}{Placeholder (id)y}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText MyPrivateInstMethod}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText MySubInstMethod}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText MySubInstMethod:}{Placeholder (id)obj}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType id}{TypedText protocolInstanceMethod:}{Placeholder (int)value}
// CHECK-CCG: ObjCInstanceMethodDecl:{ResultType int}{TypedText secondProtocolInstanceMethod}
// RUN: c-index-test -code-completion-at=%s:121:14 %s | FileCheck -check-prefix=CHECK-CCG %s
// RUN: c-index-test -code-completion-at=%s:122:7 %s | FileCheck -check-prefix=CHECK-CCH %s
// CHECK-CCH: ObjCClassMethodDecl:{ResultType id}{TypedText categoryClassMethod}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText classMethod1:}{Placeholder (id)a}{HorizontalSpace  }{Text withKeyword:}{Placeholder (int)b}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType void}{TypedText classMethod2}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText Method}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (int)i}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText Method:}{Placeholder (float)f}{HorizontalSpace  }{Text SomeArg:}{Placeholder (int)i1}{HorizontalSpace  }{Text OtherArg:}{Placeholder (id)obj}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText MyClassMethod:}{Placeholder (id)obj}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText MyPrivateMethod}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText MySubClassMethod}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText MySubPrivateMethod}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType id}{TypedText new}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType int}{TypedText OtherMethod:}{Placeholder (float)f}{HorizontalSpace  }{Text Arg1:}{Placeholder (int)i1}{HorizontalSpace  }{Text Arg2:}{Placeholder (int)i2}
// CHECK-CCH: ObjCClassMethodDecl:{ResultType id}{TypedText protocolClassMethod}
// RUN: c-index-test -code-completion-at=%s:134:6 %s | FileCheck -check-prefix=CHECK-CCI %s
// CHECK-CCI: ObjCInstanceMethodDecl:{ResultType void}{TypedText method1} (22)
// CHECK-CCI: ObjCInstanceMethodDecl:{ResultType void}{TypedText method2} (20)
