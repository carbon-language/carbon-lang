// This test is line- and column-sensitive. See below for run lines.


@interface A
- instanceMethod:(int)value withOther:(int)other;
+ classMethod;
@end

@interface B : A
@end

@implementation B
- someMethod:(A*)a {
  [a classMethod];
  [A classMethod];
  [a instanceMethod:0 withOther:1];
  [self someMethod:a];
  [super instanceMethod];
}

@end

// RUN: c-index-test -code-completion-at=%s:14:6 -std=c++11 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: ObjCInstanceMethodDecl:{ResultType id}{TypedText instanceMethod:}{Placeholder (int)}{HorizontalSpace  }{TypedText withOther:}{Placeholder (int)} (35) (parent: ObjCInterfaceDecl 'A')

// RUN: c-index-test -code-completion-at=%s:15:6 -std=c++11 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCClassMethodDecl:{ResultType id}{TypedText classMethod} (35) (parent: ObjCInterfaceDecl 'A')

// RUN: c-index-test -code-completion-at=%s:16:4 -std=c++11 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: ObjCInterfaceDecl:{TypedText A} (50) (parent: TranslationUnit '(null)')
// CHECK-CC3: ParmDecl:{ResultType A *}{TypedText a} (34)
// CHECK-CC3: ObjCInterfaceDecl:{TypedText B} (50) (parent: TranslationUnit '(null)')
// CHECK-CC3: TypedefDecl:{TypedText Class} (50) (parent: TranslationUnit '(null)')


// RUN: c-index-test -code-completion-at=%s:16:21 -x objective-c++ -std=c++11 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: NotImplemented:{ResultType B *}{TypedText self} (34)
// CHECK-CC4: NotImplemented:{ResultType A *}{TypedText super} (40)

// RUN: c-index-test -code-completion-at=%s:18:10 -x objective-c++ -std=c++11 %s | FileCheck -check-prefix=CHECK-CC1 %s
