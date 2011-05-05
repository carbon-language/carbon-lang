/* Note: the RUN lines are near the end of the file, since line/column
   matter for this test. */

@protocol MyProtocol
@property float ProtoProp;
@end

@interface Super {
  int SuperIVar;
}
@end
@interface Int : Super<MyProtocol>
{
  int IVar;
}

@property int prop1;
@end

void test_props(Int* ptr) {
  ptr.prop1 = 0;
  ptr->IVar = 0;
}

@interface Sub : Int 
@property int myProp;

- (int)myProp;
- (int)myOtherPropLikeThing;
- (int)myOtherNonPropThing:(int)value;
@end

int test_more_props(Sub *s) {
  return s.myOtherPropLikeThing;
}

// RUN: c-index-test -code-completion-at=%s:21:7 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{ResultType int}{TypedText prop1}
// CHECK-CC1: ObjCPropertyDecl:{ResultType float}{TypedText ProtoProp}
// RUN: c-index-test -code-completion-at=%s:22:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCIvarDecl:{ResultType int}{TypedText IVar} (35)
// CHECK-CC2: ObjCIvarDecl:{ResultType int}{TypedText SuperIVar} (37)
// RUN: c-index-test -code-completion-at=%s:34:12 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{ResultType int}{TypedText myOtherPropLikeThing} (37)
// CHECK-CC3: ObjCPropertyDecl:{ResultType int}{TypedText myProp} (35)
// CHECK-CC3: ObjCPropertyDecl:{ResultType int}{TypedText prop1} (35)
// CHECK-CC3: ObjCPropertyDecl:{ResultType float}{TypedText ProtoProp} (35)
