// Note: the run lines follow their respective tests, since line/column
// matter in this test.

template<class T> void (&foo(T))(T);
template<class T> void (*bar(T))(T);

int main() {
  foo(42)(42);
  bar(42)(42);
}

// RUN: c-index-test -code-completion-at=%s:8:11 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: OverloadCandidate:{Text void}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC1: Completion contexts:
// CHECK-CC1-NEXT: Any type
// CHECK-CC1-NEXT: Any value
// CHECK-CC1-NEXT: Enum tag
// CHECK-CC1-NEXT: Union tag
// CHECK-CC1-NEXT: Struct tag
// CHECK-CC1-NEXT: Class name
// CHECK-CC1-NEXT: Nested name specifier
// CHECK-CC1-NEXT: Objective-C interface

// RUN: c-index-test -code-completion-at=%s:9:11 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: OverloadCandidate:{Text void}{LeftParen (}{CurrentParameter int}{RightParen )} (1)
// CHECK-CC2: Completion contexts:
// CHECK-CC2-NEXT: Any type
// CHECK-CC2-NEXT: Any value
// CHECK-CC2-NEXT: Enum tag
// CHECK-CC2-NEXT: Union tag
// CHECK-CC2-NEXT: Struct tag
// CHECK-CC2-NEXT: Class name
// CHECK-CC2-NEXT: Nested name specifier
// CHECK-CC2-NEXT: Objective-C interface
