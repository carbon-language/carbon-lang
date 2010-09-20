typedef signed char BOOL;
#define YES ((BOOL)1)
#define NO ((BOOL)0)
#define bool _Bool
@interface A
- (int)method:(id)param1;

@property int prop1;
@end

@implementation A
- (int)method:(id)param1 {
  
}
@end

// RUN: c-index-test -code-completion-at=%s:13:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{ResultType SEL}{TypedText _cmd} (80)
// CHECK-CC1: TypedefDecl:{TypedText BOOL} (50)
// CHECK-CC1: macro definition:{TypedText bool} (51)
// CHECK-CC1: macro definition:{TypedText NO} (65)
// CHECK-CC1: NotImplemented:{ResultType A *}{TypedText self} (8)
// CHECK-CC1: macro definition:{TypedText YES} (65)
