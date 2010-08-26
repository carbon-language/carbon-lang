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

@interface C
- (int)first:(int)x second:(float)y third:(double)z;
- (id)first:(int)xx second2:(float)y2 third:(double)z;
- (void*)first:(int)xxx second3:(float)y3 third:(double)z;
@end

@interface D
- (int)first:(int)x second2:(float)y third:(double)z;
@end

@implementation D
- (int)first:(int)x second2:(float)y third:(double)z { }
@end

@interface Passing
- (oneway void)method:(in id x);
@end

// RUN: c-index-test -code-completion-at=%s:17:3 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText abc}
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText getInt}
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText getSelf}
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC1: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: c-index-test -code-completion-at=%s:17:7 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCInstanceMethodDecl:{TypedText abc}
// CHECK-CC2-NEXT: ObjCInstanceMethodDecl:{TypedText getSelf}
// CHECK-CC2: ObjCInstanceMethodDecl:{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC2: ObjCInstanceMethodDecl:{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: c-index-test -code-completion-at=%s:24:7 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText abc}
// CHECK-CC3-NEXT: ObjCInstanceMethodDecl:{TypedText getSelf}
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText init}
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}
// CHECK-CC3: ObjCInstanceMethodDecl:{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}
// RUN: c-index-test -code-completion-at=%s:33:3 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText abc}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (32)
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText getInt}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText getSecondValue}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace  }{Text return}{HorizontalSpace  }{Placeholder expression}{SemiColon ;}{VerticalSpace  }{RightBrace }} (30)
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC4: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText setValue}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// RUN: c-index-test -code-completion-at=%s:33:8 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText getInt}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText getSecondValue}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC5-NOT: {TypedText getSelf}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// CHECK-CC5: ObjCInstanceMethodDecl:{TypedText setValue}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace
// RUN: c-index-test -code-completion-at=%s:37:7 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText abc}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// CHECK-CC6-NOT: getSelf
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText initWithInt}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// CHECK-CC6: ObjCInstanceMethodDecl:{TypedText initWithTwoInts}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text y}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// RUN: c-index-test -code-completion-at=%s:42:3 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: ObjCInstanceMethodDecl:{LeftParen (}{Text id}{RightParen )}{TypedText categoryFunction}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{LeftBrace {}{VerticalSpace 
// RUN: c-index-test -code-completion-at=%s:52:21 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType id}{Informative first:}{TypedText second2:}{Text (float)y2}{HorizontalSpace  }{Text third:}{Text (double)z} (20)
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType void *}{Informative first:}{TypedText second3:}{Text (float)y3}{HorizontalSpace  }{Text third:}{Text (double)z} (20)
// CHECK-CC8: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{TypedText second:}{Text (float)y}{HorizontalSpace  }{Text third:}{Text (double)z} (5)
// RUN: c-index-test -code-completion-at=%s:52:19 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9: NotImplemented:{TypedText x} (30)
// CHECK-CC9: NotImplemented:{TypedText xx} (30)
// CHECK-CC9: NotImplemented:{TypedText xxx} (30)
// RUN: c-index-test -code-completion-at=%s:52:36 -Xclang -code-completion-patterns %s | FileCheck -check-prefix=CHECK-CCA %s
// CHECK-CCA: NotImplemented:{TypedText y2} (30)
// RUN: c-index-test -code-completion-at=%s:56:3 %s | FileCheck -check-prefix=CHECK-CCB %s
// CHECK-CCB: ObjCInstanceMethodDecl:{LeftParen (}{Text int}{RightParen )}{TypedText first}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second2}{Colon :}{LeftParen (}{Text float}{RightParen )}{Text y}{HorizontalSpace  }{Text third}{Colon :}{LeftParen (}{Text double}{RightParen )}{Text z} (30)
// RUN: c-index-test -code-completion-at=%s:56:8 %s | FileCheck -check-prefix=CHECK-CCC %s
// CHECK-CCC: ObjCInstanceMethodDecl:{TypedText first}{Colon :}{LeftParen (}{Text int}{RightParen )}{Text x}{HorizontalSpace  }{Text second2}{Colon :}{LeftParen (}{Text float}{RightParen )}{Text y}{HorizontalSpace  }{Text third}{Colon :}{LeftParen (}{Text double}{RightParen )}{Text z} (30)
// RUN: c-index-test -code-completion-at=%s:56:21 %s | FileCheck -check-prefix=CHECK-CCD %s
// FIXME: These results could be more precise.
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType id}{Informative first:}{TypedText second2:}{Text (float)y2}{HorizontalSpace  }{Text third:}{Text (double)z} (20)
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{TypedText second2:}{Text (float)y}{HorizontalSpace  }{Text third:}{Text (double)z} (5)
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType void *}{Informative first:}{TypedText second3:}{Text (float)y3}{HorizontalSpace  }{Text third:}{Text (double)z} (20)
// CHECK-CCD: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{TypedText second:}{Text (float)y}{HorizontalSpace  }{Text third:}{Text (double)z} (5)
// RUN: c-index-test -code-completion-at=%s:56:38 %s | FileCheck -check-prefix=CHECK-CCE %s
// CHECK-CCE: ObjCInstanceMethodDecl:{ResultType id}{Informative first:}{Informative second2:}{TypedText third:}{Text (double)z} (20)
// CHECK-CCE: ObjCInstanceMethodDecl:{ResultType int}{Informative first:}{Informative second2:}{TypedText third:}{Text (double)z} (5)
// RUN: c-index-test -code-completion-at=%s:60:4 %s | FileCheck -check-prefix=CHECK-CCF %s
// CHECK-CCF: ObjCInterfaceDecl:{TypedText A} (65)
// CHECK-CCF: ObjCInterfaceDecl:{TypedText B} (65)
// CHECK-CCF: NotImplemented:{TypedText bycopy} (30)
// CHECK-CCF: NotImplemented:{TypedText byref} (30)
// CHECK-CCF: NotImplemented:{TypedText in} (30)
// CHECK-CCF: NotImplemented:{TypedText inout} (30)
// CHECK-CCF: NotImplemented:{TypedText oneway} (30)
// CHECK-CCF: NotImplemented:{TypedText out} (30)
// CHECK-CCF: NotImplemented:{TypedText unsigned} (65)
// CHECK-CCF: NotImplemented:{TypedText void} (65)
// CHECK-CCF: NotImplemented:{TypedText volatile} (65)
// RUN: c-index-test -code-completion-at=%s:60:11 %s | FileCheck -check-prefix=CHECK-CCG %s
// CHECK-CCG: ObjCInterfaceDecl:{TypedText A} (65)
// CHECK-CCG: ObjCInterfaceDecl:{TypedText B} (65)
// CHECK-CCG-NOT: NotImplemented:{TypedText bycopy} (30)
// CHECK-CCG-NOT: NotImplemented:{TypedText byref} (30)
// CHECK-CCG: NotImplemented:{TypedText in} (30)
// CHECK-CCG: NotImplemented:{TypedText inout} (30)
// CHECK-CCG-NOT: NotImplemented:{TypedText oneway} (30)
// CHECK-CCG: NotImplemented:{TypedText out} (30)
// CHECK-CCG: NotImplemented:{TypedText unsigned} (65)
// CHECK-CCG: NotImplemented:{TypedText void} (65)
// CHECK-CCG: NotImplemented:{TypedText volatile} (65)
// RUN: c-index-test -code-completion-at=%s:60:24 %s | FileCheck -check-prefix=CHECK-CCF %s
// RUN: c-index-test -code-completion-at=%s:60:26 %s | FileCheck -check-prefix=CHECK-CCH %s
// CHECK-CCH: ObjCInterfaceDecl:{TypedText A} (65)
// CHECK-CCH: ObjCInterfaceDecl:{TypedText B} (65)
// CHECK-CCH: NotImplemented:{TypedText bycopy} (30)
// CHECK-CCH: NotImplemented:{TypedText byref} (30)
// CHECK-CCH-NOT: NotImplemented:{TypedText in} (30)
// CHECK-CCH: NotImplemented:{TypedText inout} (30)
// CHECK-CCH: NotImplemented:{TypedText oneway} (30)
// CHECK-CCH: NotImplemented:{TypedText out} (30)
// CHECK-CCH: NotImplemented:{TypedText unsigned} (65)
// CHECK-CCH: NotImplemented:{TypedText void} (65)
// CHECK-CCH: NotImplemented:{TypedText volatile} (65)
