/* The run lines are below, because this test is line- and
   column-number sensitive. */
@interface MyClass { }
- (int)myMethod:(int)arg;
@end

@implementation MyClass 
- (int)myMethod:(int)arg {
  @synchronized (@encode(MyClass)) { }
}
@end
// RUN: c-index-test -code-completion-at=%s:9:4 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC1: {TypedText protocol}{LeftParen (}{Placeholder protocol-name}{RightParen )}
// CHECK-CC1: {TypedText selector}{LeftParen (}{Placeholder selector}{RightParen )}
// CHECK-CC1: {TypedText synchronized}{Text  }{LeftParen (}{Placeholder expression}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}
// CHECK-CC1: {TypedText throw}{Text  }{Placeholder expression}{Text ;}
// CHECK-CC1: {TypedText try}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @catch}{LeftParen (}{Placeholder parameter}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @finally}{LeftBrace {}{Placeholder statements}{RightBrace }}
// RUN: c-index-test -code-completion-at=%s:9:19 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC2: {TypedText protocol}{LeftParen (}{Placeholder protocol-name}{RightParen )}
// CHECK-CC2: {TypedText selector}{LeftParen (}{Placeholder selector}{RightParen )}

