// Code-completion through the C interface
#include "nonexistent_header.h"
struct X {
  int member;
  
  enum E { Val1 };
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
  // RUN: c-index-test -code-completion-at=%s:23:11 %s | FileCheck -check-prefix=CHECK-MEMBER %s
  get_Z().member = 17;
}


float& overloaded(int i, long second);
double& overloaded(float f, int second);
int& overloaded(Z z, int second);
                
void test_overloaded() {
  // RUN: c-index-test -code-completion-at=%s:33:18 %s | FileCheck -check-prefix=CHECK-OVERLOAD %s
  overloaded(Z(), 0);
}

// CHECK-MEMBER: FieldDecl:{ResultType double}{TypedText member}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{Informative Y::}{TypedText memfunc}{LeftParen (}{Optional {Placeholder int i}}{RightParen )}
// CHECK-MEMBER: EnumConstantDecl:{ResultType enum X::E}{Informative E::}{TypedText Val1}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{Informative X::}{TypedText ~X}{LeftParen (}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{Informative Y::}{TypedText ~Y}{LeftParen (}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{TypedText ~Z}{LeftParen (}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType int}{TypedText operator int}{LeftParen (}{RightParen )}{Informative  const}
// CHECK-MEMBER: FunctionDecl:{ResultType struct Z &}{TypedText operator=}{LeftParen (}{Placeholder struct Z const &}{RightParen )}
// CHECK-MEMBER: FieldDecl:{ResultType int}{Text X::}{TypedText member}
// CHECK-MEMBER: FieldDecl:{ResultType float}{Text Y::}{TypedText member}
// CHECK-MEMBER: FunctionDecl:{ResultType struct X &}{Text X::}{TypedText operator=}{LeftParen (}{Placeholder struct X const &}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType struct Y &}{Text Y::}{TypedText operator=}{LeftParen (}{Placeholder struct Y const &}{RightParen )}
// CHECK-MEMBER: StructDecl:{TypedText X}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Y}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Z}{Text ::}

// CHECK-OVERLOAD: NotImplemented:{ResultType int &}{Text overloaded}{LeftParen (}{Text struct Z z}{Comma , }{CurrentParameter int second}{RightParen )}
// CHECK-OVERLOAD: NotImplemented:{ResultType float &}{Text overloaded}{LeftParen (}{Text int i}{Comma , }{CurrentParameter long second}{RightParen )}
// CHECK-OVERLOAD: NotImplemented:{ResultType double &}{Text overloaded}{LeftParen (}{Text float f}{Comma , }{CurrentParameter int second}{RightParen )}
