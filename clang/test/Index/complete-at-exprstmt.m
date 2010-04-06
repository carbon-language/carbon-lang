/* The run lines are below, because this test is line- and
   column-number sensitive. */
@interface MyClass { int ivar; }
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
// CHECK-CC1: {TypedText synchronized}{HorizontalSpace  }{LeftParen (}{Placeholder expression}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}
// CHECK-CC1: {TypedText throw}{HorizontalSpace  }{Placeholder expression}
// CHECK-CC1: {TypedText try}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @catch}{LeftParen (}{Placeholder parameter}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @finally}{LeftBrace {}{Placeholder statements}{RightBrace }}
// RUN: c-index-test -code-completion-at=%s:9:19 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC2: {TypedText protocol}{LeftParen (}{Placeholder protocol-name}{RightParen )}
// CHECK-CC2: {TypedText selector}{LeftParen (}{Placeholder selector}{RightParen )}
// RUN: c-index-test -code-completion-at=%s:9:3 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: NotImplemented:{TypedText @encode}{LeftParen (}{Placeholder type-name}{RightParen )}
// CHECK-CC3: NotImplemented:{TypedText @protocol}{LeftParen (}{Placeholder protocol-name}{RightParen )}
// CHECK-CC3: NotImplemented:{TypedText @selector}{LeftParen (}{Placeholder selector}{RightParen )}
// CHECK-CC3: NotImplemented:{TypedText @synchronized}{HorizontalSpace  }{LeftParen (}{Placeholder expression}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}
// CHECK-CC3: NotImplemented:{TypedText @throw}{HorizontalSpace  }{Placeholder expression}
// CHECK-CC3: NotImplemented:{TypedText @try}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @catch}{LeftParen (}{Placeholder parameter}{RightParen )}{LeftBrace {}{Placeholder statements}{RightBrace }}{Text @finally}{LeftBrace {}{Placeholder statements}{RightBrace }}
// CHECK-CC3: NotImplemented:{ResultType SEL}{TypedText _cmd}
// CHECK-CC3: ParmDecl:{ResultType int}{TypedText arg}
// CHECK-CC3: TypedefDecl:{TypedText Class}
// CHECK-CC3: TypedefDecl:{TypedText id}
// CHECK-CC3: ObjCIvarDecl:{ResultType int}{TypedText ivar}
// CHECK-CC3: ObjCInterfaceDecl:{TypedText MyClass}
// CHECK-CC3: TypedefDecl:{TypedText SEL}
// CHECK-CC3: NotImplemented:{ResultType MyClass *}{TypedText self}
