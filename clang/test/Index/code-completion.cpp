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
struct W { };
struct Z get_Z();
namespace N { }
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

Z::operator int() const {
  return 0;
}

// CHECK-MEMBER: FieldDecl:{ResultType double}{TypedText member}
// CHECK-MEMBER: FieldDecl:{ResultType int}{Text X::}{TypedText member}
// CHECK-MEMBER: FieldDecl:{ResultType float}{Text Y::}{TypedText member}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{Informative Y::}{TypedText memfunc}{LeftParen (}{Optional {Placeholder int i}}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType int}{TypedText operator int}{LeftParen (}{RightParen )}{Informative  const}
// CHECK-MEMBER: FunctionDecl:{ResultType Z &}{TypedText operator=}{LeftParen (}{Placeholder Z const &}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType X &}{Text X::}{TypedText operator=}{LeftParen (}{Placeholder X const &}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType Y &}{Text Y::}{TypedText operator=}{LeftParen (}{Placeholder Y const &}{RightParen )}
// CHECK-MEMBER: EnumConstantDecl:{ResultType X::E}{Informative E::}{TypedText Val1}
// CHECK-MEMBER: StructDecl:{TypedText X}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Y}{Text ::}
// CHECK-MEMBER: StructDecl:{TypedText Z}{Text ::}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{Informative X::}{TypedText ~X}{LeftParen (}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{Informative Y::}{TypedText ~Y}{LeftParen (}{RightParen )}
// CHECK-MEMBER: FunctionDecl:{ResultType void}{TypedText ~Z}{LeftParen (}{RightParen )}

// CHECK-OVERLOAD: NotImplemented:{ResultType int &}{Text overloaded}{LeftParen (}{Text Z z}{Comma , }{CurrentParameter int second}{RightParen )}
// CHECK-OVERLOAD: NotImplemented:{ResultType float &}{Text overloaded}{LeftParen (}{Text int i}{Comma , }{CurrentParameter long second}{RightParen )}
// CHECK-OVERLOAD: NotImplemented:{ResultType double &}{Text overloaded}{LeftParen (}{Text float f}{Comma , }{CurrentParameter int second}{RightParen )}

// RUN: c-index-test -code-completion-at=%s:37:10 %s | FileCheck -check-prefix=CHECK-EXPR %s
// CHECK-EXPR: NotImplemented:{TypedText int} (65)
// CHECK-EXPR: NotImplemented:{TypedText long} (65)
// CHECK-EXPR: FieldDecl:{ResultType double}{TypedText member} (10)
// CHECK-EXPR: FieldDecl:{ResultType int}{Text X::}{TypedText member} (5)
// CHECK-EXPR: FieldDecl:{ResultType float}{Text Y::}{TypedText member} (11)
// CHECK-EXPR: FunctionDecl:{ResultType void}{TypedText memfunc}{LeftParen (}{Optional {Placeholder int i}}{RightParen )} (22)
// CHECK-EXPR: NotImplemented:{TypedText N}{Text ::} (75)
