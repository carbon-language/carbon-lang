// This test is line- and column-sensitive, so test commands are at the bottom.
@protocol P
- (int)method:(id)param1;
@end

@interface A <P>
- (int)method:(id)param1;

@property int prop1;
@end

@implementation A
- (int)method:(id)param1 {
  int q2;
  for(id q in param1) {
    int y;
  }
  id q;
  for(q in param1) {
    int y;
  }
}
@end

// RUN: c-index-test -code-completion-at=%s:7:19 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1-NOT: NotImplemented:{TypedText extern} (40)
// CHECK-CC1: NotImplemented:{TypedText param1} (40)
// RUN: c-index-test -code-completion-at=%s:9:15 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:15:10 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:16:9 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText const} (40)
// CHECK-CC2-NOT: int
// CHECK-CC2: NotImplemented:{TypedText restrict} (40)
// CHECK-CC2: NotImplemented:{TypedText volatile} (40)
// RUN: c-index-test -code-completion-at=%s:15:15 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ParmDecl:{ResultType id}{TypedText param1} (34)
// CHECK-CC3-NOT: VarDecl:{ResultType int}{TypedText q2}
// CHECK-CC3-NOT: VarDecl:{ResultType id}{TypedText q}
// CHECK-CC3: NotImplemented:{ResultType A *}{TypedText self} (34)
// CHECK-CC3: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
// RUN: c-index-test -code-completion-at=%s:15:15 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ParmDecl:{ResultType id}{TypedText param1} (34)
// CHECK-CC4-NOT: VarDecl:{ResultType int}{TypedText q2}
// CHECK-CC4: NotImplemented:{ResultType A *}{TypedText self} (34)
// CHECK-CC4: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
