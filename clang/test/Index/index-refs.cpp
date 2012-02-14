
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

template <typename T1, typename T2>
struct TS { };

template <typename T>
struct TS<T, int> {
  typedef int MyInt;
};

void foo3() {
  TS<int, int> s;
}

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

// CHECK:      [indexDeclaration]: kind: c++-class-template | name: TS | {{.*}} | loc: 47:8
// CHECK-NEXT: [indexDeclaration]: kind: struct-template-partial-spec | name: TS | USR: c:@SP>1#T@TS>#t0.0#I | {{.*}} | loc: 50:8
// CHECK-NEXT: [indexDeclaration]: kind: typedef | name: MyInt | USR: c:index-refs.cpp@593@SP>1#T@TS>#t0.0#I@T@MyInt | {{.*}} | loc: 51:15 | semantic-container: [TS:50:8] | lexical-container: [TS:50:8]
/* when indexing implicit instantiations
  [indexDeclaration]: kind: struct-template-spec | name: TS | USR: c:@S@TS>#I | {{.*}} | loc: 50:8
  [indexDeclaration]: kind: typedef | name: MyInt | USR: c:index-refs.cpp@593@S@TS>#I@T@MyInt | {{.*}} | loc: 51:15 | semantic-container: [TS:50:8] | lexical-container: [TS:50:8]
 */
// CHECK-NEXT: [indexDeclaration]: kind: function | name: foo3
/* when indexing implicit instantiations
  [indexEntityReference]: kind: struct-template-spec | name: TS | USR: c:@S@TS>#I | {{.*}} | loc: 55:3
*/
// CHECK-NEXT: [indexEntityReference]: kind: c++-class-template | name: TS | USR: c:@ST>2#T#T@TS | {{.*}} | loc: 55:3
