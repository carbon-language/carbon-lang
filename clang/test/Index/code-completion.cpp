// Code-completion through the C interface
#include "nonexistent_header.h"
struct X {
  int member;
};

struct Y {
  float member;
  void memfunc(int i = 17);
};

struct Z : X, Y {
  double member;
  operator int() const;
};

struct Z get_Z();

void test_Z() {
  // RUN: c-index-test -code-completion-at=%s:21:11 %s | FileCheck -check-prefix=CHECK-MEMBER %s
  get_Z().member = 17;
}


float& overloaded(int i, long second);
double& overloaded(float f, int second);
int& overloaded(Z z, int second);
                
void test_overloaded() {
  // RUN: c-index-test -code-completion-at=%s:31:18 %s | FileCheck -check-prefix=CHECK-OVERLOAD %s
  overloaded(Z(), 0);
}

// CHECK-MEMBER: FieldDecl:{TypedText member}
// CHECK-MEMBER: FunctionDecl:{Informative Y::}{TypedText memfunc}{LeftParen (}{Optional {Placeholder int i}}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{Informative X::}{TypedText ~X}{LeftParen (}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{TypedText operator int}{LeftParen (}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{TypedText operator=}{LeftParen (}{Placeholder struct Z const &}{RightParen )}
// CHECK-MEMBER: StructDecl:{TypedText X}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Y}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Z}{Text ::}
// CHECK-MEMBER: FieldDecl:{Text X::}{TypedText member}
// CHECK-MEMBER: FieldDecl:{Text Y::}{TypedText member}
// CHECK-MEMBER: FunctionDecl:{Text X::}{TypedText operator=}{LeftParen (}{Placeholder struct X const &}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{Text Y::}{TypedText operator=}{LeftParen (}{Placeholder struct Y const &}{RightParen )}

// CHECK-OVERLOAD: NotImplemented:{Text overloaded}{LeftParen (}{Text struct Z z}{Comma , }{CurrentParameter int second}{RightParen )}
// CHECK-OVERLOAD: NotImplemented:{Text overloaded}{LeftParen (}{Text int i}{Comma , }{CurrentParameter long second}{RightParen )}
// CHECK-OVERLOAD: NotImplemented:{Text overloaded}{LeftParen (}{Text float f}{Comma , }{CurrentParameter int second}{RightParen )}
