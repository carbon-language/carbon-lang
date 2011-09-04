class MyCls {
  void in_foo() {
    vec.x = 0;
  }
  void out_foo();

  struct Vec { int x, y; };
  Vec vec;
};

void MyCls::out_foo() {
  vec.x = 0;
}

// RUN: c-index-test -code-completion-at=%s:3:9 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:12:7 %s | FileCheck %s
// CHECK:      CXXMethod:{ResultType MyCls::Vec &}{TypedText operator=}{LeftParen (}{Placeholder const MyCls::Vec &}{RightParen )} (34)
// CHECK-NEXT: StructDecl:{TypedText Vec}{Text ::} (75)
// CHECK-NEXT: FieldDecl:{ResultType int}{TypedText x} (35)
// CHECK-NEXT: FieldDecl:{ResultType int}{TypedText y} (35)
// CHECK-NEXT: CXXDestructor:{ResultType void}{TypedText ~Vec}{LeftParen (}{RightParen )} (34)
// CHECK-NEXT: Completion contexts:
// CHECK-NEXT: Dot member access
// CHECK-NEXT: Container Kind: StructDecl
