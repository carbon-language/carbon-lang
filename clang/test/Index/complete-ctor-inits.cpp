// The run lines are below, because this test is line- and
// column-number sensitive.

template<typename T>
struct X {
  X(T);
};

struct Virt { };
struct Y : virtual Virt { };

struct Z : public X<int>, public Y {
  Z();

  int a, b, c;
};

Z::Z() : ::X<int>(0), Virt(), a() { }

// RUN: c-index-test -code-completion-at=%s:18:10 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: NotImplemented:{TypedText a}{LeftParen (}{Placeholder args}{RightParen )} (4)
// CHECK-CC1: NotImplemented:{TypedText b}{LeftParen (}{Placeholder args}{RightParen )} (5)
// CHECK-CC1: NotImplemented:{TypedText c}{LeftParen (}{Placeholder args}{RightParen )} (6)
// CHECK-CC1: NotImplemented:{TypedText Virt}{LeftParen (}{Placeholder args}{RightParen )} (3)
// CHECK-CC1: NotImplemented:{TypedText X<int>}{LeftParen (}{Placeholder args}{RightParen )} (1)
// CHECK-CC1: NotImplemented:{TypedText Y}{LeftParen (}{Placeholder args}{RightParen )} (2)

// RUN: c-index-test -code-completion-at=%s:18:23 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: NotImplemented:{TypedText a}{LeftParen (}{Placeholder args}{RightParen )} (3)
// CHECK-CC2: NotImplemented:{TypedText b}{LeftParen (}{Placeholder args}{RightParen )} (4)
// CHECK-CC2: NotImplemented:{TypedText c}{LeftParen (}{Placeholder args}{RightParen )} (5)
// CHECK-CC2: NotImplemented:{TypedText Virt}{LeftParen (}{Placeholder args}{RightParen )} (2)
// CHECK-CC2: NotImplemented:{TypedText Y}{LeftParen (}{Placeholder args}{RightParen )} (1)
