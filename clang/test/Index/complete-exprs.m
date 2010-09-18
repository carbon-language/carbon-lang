@interface A
- (int)method:(id)param1;

@property int prop1;
@end

@implementation A
- (int)method:(id)param1 {
  
}
@end

// RUN: c-index-test -code-completion-at=%s:9:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{ResultType SEL}{TypedText _cmd} (80)
// CHECK-CC1: NotImplemented:{ResultType A *}{TypedText self} (8)
