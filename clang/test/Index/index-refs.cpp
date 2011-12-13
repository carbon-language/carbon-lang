
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
