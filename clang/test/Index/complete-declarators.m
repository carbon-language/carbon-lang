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
  for(id x in param1) {
    int y;
  }
}
@end

// RUN: c-index-test -code-completion-at=%s:7:19 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1-NOT: NotImplemented:{TypedText extern} (30)
// CHECK-CC1: NotImplemented:{TypedText param1} (30)
// RUN: c-index-test -code-completion-at=%s:9:15 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:14:10 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: c-index-test -code-completion-at=%s:15:9 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText const} (30)
// CHECK-CC2-NOT: int
// CHECK-CC2: NotImplemented:{TypedText restrict} (30)
// CHECK-CC2: NotImplemented:{TypedText volatile} (30)
