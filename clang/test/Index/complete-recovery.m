/* Run lines are at the end, since line/column matter in this test. */

@interface A
- (void)method:(int)x;
@end

@implementation A
- (void)method:(int)x {
  A *a = [A method:1];
  blarg * blah = wibble
}
@end

// RUN: c-index-test -code-completion-at=%s:9:20 -Xclang -code-completion-patterns %s 2>%t | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: not grep error %t
// CHECK-CC1: NotImplemented:{TypedText @encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC1-NOT: NotImplemented:{TypedText _Bool}
// CHECK-CC1: VarDecl:{ResultType A *}{TypedText a}
// CHECK-CC1: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )}

// RUN: c-index-test -code-completion-at=%s:10:24 -Xclang -code-completion-patterns %s 2>%t | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText @encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC2: NotImplemented:{TypedText _Bool}
// CHECK-CC2: VarDecl:{ResultType A *}{TypedText a}
// CHECK-CC2: NotImplemented:{TypedText sizeof}{LeftParen (}{Placeholder expression-or-type}{RightParen )}
