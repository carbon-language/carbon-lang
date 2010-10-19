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
  
  for(BOOL B = YES; ; ) { }
}
@end

// RUN: c-index-test -code-completion-at=%s:13:2 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{ResultType SEL}{TypedText _cmd} (80)
// CHECK-CC1: TypedefDecl:{TypedText BOOL} (50)
// CHECK-CC1: macro definition:{TypedText bool} (51)
// CHECK-CC1: macro definition:{TypedText NO} (65)
// CHECK-CC1: NotImplemented:{ResultType A *}{TypedText self} (34)
// CHECK-CC1: macro definition:{TypedText YES} (65)
// RUN: c-index-test -code-completion-at=%s:14:7 %s | FileCheck -check-prefix=CHECK-CC2 %s
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_CACHING=1 c-index-test -code-completion-at=%s:14:7 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: TypedefDecl:{TypedText BOOL} (50)
// CHECK-CC2: NotImplemented:{TypedText char} (50)
// CHECK-CC2: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )} (40)
