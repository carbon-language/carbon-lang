// The run lines are below, because this test is line- and
// column-number sensitive.

struct A {
  virtual void foo(int x, int y);
  virtual void bar(double x);
  virtual void bar(float x);
};

struct B : A {
  void foo(int a, int b);
  void bar(float real);
};

void B::foo(int a, int b) {
  A::foo(a, b);
}

void B::bar(float real) {
  A::bar(real);
}

// RUN: c-index-test -code-completion-at=%s:16:3 %s | FileCheck -check-prefix=CHECK-FOO-UNQUAL %s
// CHECK-FOO-UNQUAL: CXXMethod:{Text A::}{TypedText foo}{LeftParen (}{Placeholder a}{Comma , }{Placeholder b}{RightParen )} (20)

// RUN: c-index-test -code-completion-at=%s:20:3 %s | FileCheck -check-prefix=CHECK-BAR-UNQUAL %s
// CHECK-BAR-UNQUAL: CXXMethod:{Text A::}{TypedText bar}{LeftParen (}{Placeholder real}{RightParen )} (20)
// CHECK-BAR-UNQUAL: CXXMethod:{ResultType void}{TypedText bar}{LeftParen (}{Placeholder float real}{RightParen )} (34)
// CHECK-BAR-UNQUAL: CXXMethod:{ResultType void}{Text A::}{TypedText bar}{LeftParen (}{Placeholder double x}{RightParen )} (36)

// RUN: c-index-test -code-completion-at=%s:16:6 %s | FileCheck -check-prefix=CHECK-FOO-QUAL %s
// CHECK-FOO-QUAL: CXXMethod:{TypedText foo}{LeftParen (}{Placeholder a}{Comma , }{Placeholder b}{RightParen )} (20)

