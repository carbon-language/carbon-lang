// RUN: %clang_cc1 -fsyntax-only -pedantic -std=c++0x -verify -triple x86_64-apple-darwin %s

enum class E1 {
  Val1 = 1L
};

enum struct E2 {
  Val1 = '\0'
};

E1 v1 = Val1; // expected-error{{undeclared identifier}}
E1 v2 = E1::Val1;

static_assert(sizeof(E1) == sizeof(int), "bad size");
static_assert(sizeof(E1::Val1) == sizeof(int), "bad size");
static_assert(sizeof(E2) == sizeof(int), "bad size");
static_assert(sizeof(E2::Val1) == sizeof(int), "bad size");

E1 v3 = E2::Val1; // expected-error{{cannot initialize a variable}}
int x1 = E1::Val1; // expected-error{{cannot initialize a variable}}

enum E3 : char {
  Val2 = 1
};

E3 v4 = Val2;
E1 v5 = Val2; // expected-error{{cannot initialize a variable}}

static_assert(sizeof(E3) == 1, "bad size");

int x2 = Val2;

int a1[Val2];
int a2[E1::Val1]; // expected-error{{size of array has non-integer type}}

int* p1 = new int[Val2];
int* p2 = new int[E1::Val1]; // FIXME Expected-error{{must have integral}}

enum class E4 {
  e1 = -2147483648, // ok
  e2 = 2147483647, // ok
  e3 = 2147483648 // expected-error{{value is not representable}}
};

enum class E5 {
  e1 = 2147483647, // ok
  e2 // expected-error{{2147483648 is not representable in the underlying}}
};

enum class E6 : bool {
    e1 = false, e2 = true,
    e3 // expected-error{{2 is not representable in the underlying}}
};

enum E7 : bool {
    e1 = false, e2 = true,
    e3 // expected-error{{2 is not representable in the underlying}}
};

template <class T>
struct X {
  enum E : T {
    e1, e2,
    e3 // expected-error{{2 is not representable in the underlying}}
  };
};

X<bool> X2; // expected-note{{in instantiation of template}}

enum Incomplete1; // expected-error{{C++ forbids forward references}}

enum Complete1 : int;
Complete1 complete1;

enum class Complete2;
Complete2 complete2;

// All the redeclarations below are done twice on purpose. Tests that the type
// of the declaration isn't changed.

enum class Redeclare2; // expected-note{{previous use is here}} expected-note{{previous use is here}}
enum Redeclare2; // expected-error{{previously declared as scoped}}
enum Redeclare2; // expected-error{{previously declared as scoped}}

enum Redeclare3 : int; // expected-note{{previous use is here}} expected-note{{previous use is here}}
enum Redeclare3; // expected-error{{previously declared with fixed underlying type}}
enum Redeclare3; // expected-error{{previously declared with fixed underlying type}}

enum class Redeclare5;
enum class Redeclare5 : int; // ok

enum Redeclare6 : int; // expected-note{{previous use is here}} expected-note{{previous use is here}}
enum Redeclare6 : short; // expected-error{{redeclared with different underlying type}}
enum Redeclare6 : short; // expected-error{{redeclared with different underlying type}}

enum class Redeclare7; // expected-note{{previous use is here}} expected-note{{previous use is here}}
enum class Redeclare7 : short; // expected-error{{redeclared with different underlying type}}
enum class Redeclare7 : short; // expected-error{{redeclared with different underlying type}}

enum : long {
  long_enum_val = 10000
};

enum : long x; // expected-error{{unnamed enumeration must be a definition}} \
// expected-warning{{declaration does not declare anything}}
