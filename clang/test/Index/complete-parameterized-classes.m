@protocol NSObject
@end

@interface NSObject
@end

@interface Test<T : id, U : NSObject *> : NSObject
{
@public
	U myVar;
}
-(U)getit:(T)val;
-(void)apply:(void(^)(T, U))block;

@property (strong) T prop;
@end

@interface MyClsA : NSObject
@end
@interface MyClsB : NSObject
@end

void test1(Test<MyClsA*, MyClsB*> *obj) {
  [obj ];
  obj.;
  obj->;
}

void test2(Test *obj) {
  [obj ];
  obj.;
  obj->;
}

// RUN: c-index-test -code-completion-at=%s:24:8 %s | FileCheck -check-prefix=CHECK-CC0 %s
// CHECK-CC0: ObjCInstanceMethodDecl:{ResultType void}{TypedText apply:}{Placeholder ^(MyClsA *, MyClsB *)block} (35)
// CHECK-CC0: ObjCInstanceMethodDecl:{ResultType MyClsB *}{TypedText getit:}{Placeholder (MyClsA *)} (35)

// RUN: c-index-test -code-completion-at=%s:25:7 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{ResultType MyClsA *}{TypedText prop} (35)

// RUN: c-index-test -code-completion-at=%s:26:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCIvarDecl:{ResultType MyClsB *}{TypedText myVar} (35)

// RUN: c-index-test -code-completion-at=%s:30:8 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType void}{TypedText apply:}{Placeholder ^(id, NSObject *)block} (35)
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType __kindof NSObject *}{TypedText getit:}{Placeholder (id)} (35)

// RUN: c-index-test -code-completion-at=%s:31:7 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCPropertyDecl:{ResultType id}{TypedText prop} (35)

// RUN: c-index-test -code-completion-at=%s:32:8 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCIvarDecl:{ResultType __kindof NSObject *}{TypedText myVar} (35)
