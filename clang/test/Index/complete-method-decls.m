/* Note: the RUN lines are near the end of the file, since line/column
   matter for this test. */

@protocol P1
- (id)abc;
- (id)initWithInt:(int)x;
- (id)initWithTwoInts:(int)x second:(int)y;
- (int)getInt;
- (id)getSelf;
@end

@protocol P2<P1>
+ (id)alloc;
@end

@interface A <P1>
- (id)init;
- (int)getValue;
@end

@interface B : A<P2>
- (id)initWithInt:(int)x;
- (int)getSecondValue;
- (id)getSelf;
- (int)setValue:(int)x;
@end

@interface B (FooBar)
- (id)categoryFunction:(int)x;
@end

@implementation B
- (int)getSecondValue { return 0; }
- (id)init { return self; }
- (id)getSelf { return self; }
- (void)setValue:(int)x { }
- (id)initWithTwoInts:(int)x second:(int)y { return self; }
+ (id)alloc { return 0; }
@end

@implementation B (FooBar)
- (id)categoryFunction:(int)x { return self; }
@end

// RUN: c-index-test -code-completion-at=%s:17:3 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText abc}
// CHECK-CC1: NotImplemented:{LeftParen (}{Text int}{RightParen )}{TypedText getInt}
// CHECK-CC1: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText getSelf}
// CHECK-CC1: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC1: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: c-index-test -code-completion-at=%s:17:7 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText abc}
// CHECK-CC2-NEXT: NotImplemented:{TypedText getSelf}
// CHECK-CC2: NotImplemented:{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC2: NotImplemented:{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: c-index-test -code-completion-at=%s:24:7 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: NotImplemented:{TypedText abc}
// CHECK-CC3-NEXT: NotImplemented:{TypedText getSelf}
// CHECK-CC3: NotImplemented:{TypedText init}
// CHECK-CC3: NotImplemented:{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC3: NotImplemented:{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: c-index-test -code-completion-at=%s:33:3 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText abc}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: NotImplemented:{LeftParen (}{Text int}{RightParen )}{TypedText getInt}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: NotImplemented:{LeftParen (}{Text int}{RightParen )}{TypedText getSecondValue}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: NotImplemented:{LeftParen (}{Text int}{RightParen )}{TypedText setValue}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// RUN: c-index-test -code-completion-at=%s:33:8 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: NotImplemented:{TypedText getInt}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC5: NotImplemented:{TypedText getSecondValue}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC5-NOT: {TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC5: NotImplemented:{TypedText setValue}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// RUN: c-index-test -code-completion-at=%s:37:7 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: NotImplemented:{TypedText abc}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// CHECK-CC6-NOT: getSelf
// CHECK-CC6: NotImplemented:{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// CHECK-CC6: NotImplemented:{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// RUN: c-index-test -code-completion-at=%s:42:3 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: NotImplemented:{LeftParen (}{Text id}{RightParen )}{TypedText categoryFunction}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 

