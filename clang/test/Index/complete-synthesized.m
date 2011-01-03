// Note: this test is line- and column-sensitive. Test commands are at
// the end.


@interface A
@property int prop1;
@end

@interface B : A {
  float _prop2;
}
@property float prop2;
@property short prop3;
@end

@interface B ()
@property double prop4;
@end

@implementation B
@synthesize prop2 = _prop2;

- (int)method {
  return _prop2;
}

@dynamic prop3;

- (short)method2 {
  return prop4;
}

- (short)method3 {
  return prop3;
}
@end

// RUN: c-index-test -code-completion-at=%s:24:1 -Xclang -fobjc-nonfragile-abi2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{TypedText _Bool} (50)
// CHECK-CC1: ObjCIvarDecl:{ResultType float}{TypedText _prop2} (35)
// CHECK-CC1-NOT: prop2
// CHECK-CC1: ObjCPropertyDecl:{ResultType short}{TypedText prop3} (35)
// CHECK-CC1: ObjCPropertyDecl:{ResultType double}{TypedText prop4} (35)

// RUN: c-index-test -code-completion-at=%s:30:2 -Xclang -fobjc-nonfragile-abi2 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText _Bool} (50)
// CHECK-CC2: ObjCIvarDecl:{ResultType float}{TypedText _prop2} (35)
// CHECK-CC2-NOT: prop3
// CHECK-CC2: ObjCPropertyDecl:{ResultType double}{TypedText prop4} (35)

// RUN: c-index-test -code-completion-at=%s:34:2 -Xclang -fobjc-nonfragile-abi2 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: NotImplemented:{TypedText _Bool} (50)
// CHECK-CC3: ObjCIvarDecl:{ResultType float}{TypedText _prop2} (35)
// CHECK-CC3: ObjCPropertyDecl:{ResultType double}{TypedText prop4}
// CHECK-CC3-NOT: ObjCPropertyDecl:{ResultType double}{TypedText prop4} (35)
// CHECK-CC1: restrict
