
namespace NS {
  extern int gx;
  typedef int MyInt;
}

enum {
  EnumVal = 1
};

NS::MyInt NS::gx = EnumVal;

void foo() {
  NS::MyInt x;
}

enum {
  SecondVal = EnumVal
};

struct S {
  S& operator++();
  int operator*();
  S& operator=(int x);
  S& operator!=(int x);
  S& operator()(int x);
};

void foo2(S &s) {
  (void)++s;
  (void)*s;
  s = 3;
  (void)(s != 3);
  s(3);
}

namespace NS {
  namespace Inn {}
  typedef int Foo;
}

using namespace NS;
using namespace NS::Inn;
using NS::Foo;


// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK:      [indexDeclaration]: kind: namespace | name: NS
// CHECK-NEXT: [indexDeclaration]: kind: variable | name: gx
// CHECK-NEXT: [indexDeclaration]: kind: typedef | name: MyInt
// CHECK-NEXT: [indexDeclaration]: kind: enum
// CHECK-NEXT: [indexDeclaration]: kind: enumerator | name: EnumVal
// CHECK-NEXT: [indexDeclaration]: kind: variable | name: gx
// CHECK-NEXT: [indexEntityReference]: kind: namespace | name: NS
// CHECK-NEXT: [indexEntityReference]: kind: typedef | name: MyInt
// CHECK-NEXT: [indexEntityReference]: kind: namespace | name: NS
// CHECK-NEXT: [indexEntityReference]: kind: enumerator | name: EnumVal
// CHECK-NEXT: [indexDeclaration]: kind: function | name: foo
// CHECK-NEXT: [indexEntityReference]: kind: namespace | name: NS
// CHECK-NEXT: [indexEntityReference]: kind: typedef | name: MyInt
// CHECK-NEXT: [indexDeclaration]: kind: enum
// CHECK-NEXT: [indexDeclaration]: kind: enumerator | name: SecondVal
// CHECK-NEXT: [indexEntityReference]: kind: enumerator | name: EnumVal

// CHECK:      [indexDeclaration]: kind: function | name: foo2
// CHECK:      [indexEntityReference]: kind: c++-instance-method | name: operator++
// CHECK-NEXT: [indexEntityReference]: kind: c++-instance-method | name: operator*
// CHECK-NEXT: [indexEntityReference]: kind: c++-instance-method | name: operator=
// CHECK-NEXT: [indexEntityReference]: kind: c++-instance-method | name: operator!=
// CHECK-NEXT: [indexEntityReference]: kind: c++-instance-method | name: operator()

// CHECK:      [indexEntityReference]: kind: namespace | name: NS | {{.*}} | loc: 42:17
// CHECK-NEXT: [indexEntityReference]: kind: namespace | name: NS | {{.*}} | loc: 43:17
// CHECK-NEXT: [indexEntityReference]: kind: namespace | name: Inn | {{.*}} | loc: 43:21
// CHECK-NEXT: [indexEntityReference]: kind: namespace | name: NS | {{.*}} | loc: 44:7
// CHECK-NEXT: [indexEntityReference]: kind: typedef | name: Foo | {{.*}} | loc: 44:11
