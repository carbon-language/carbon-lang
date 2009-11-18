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

// RUN: c-index-test -code-completion-at=%s:21:7 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCPropertyDecl:{TypedText prop1}
// CHECK-CC1: ObjCPropertyDecl:{TypedText ProtoProp}
// RUN: c-index-test -code-completion-at=%s:22:8 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCIvarDecl:{TypedText IVar}
// CHECK-CC2: ObjCIvarDecl:{TypedText SuperIVar}
