// RUN: %clang_cc1 -fsyntax-only -verify %s -Winvalid-offsetof

struct NonPOD {
  virtual void f();
  int m;
};

struct P {
  NonPOD fieldThatPointsToANonPODType;
};

void f() {
  int i = __builtin_offsetof(P, fieldThatPointsToANonPODType.m); // expected-warning{{offset of on non-POD type 'P'}}
}

struct Base { int x; };
struct Derived : Base { int y; };
int o = __builtin_offsetof(Derived, x); // expected-warning{{offset of on non-POD type}}

const int o2 = sizeof(__builtin_offsetof(Derived, x));

struct HasArray {
  int array[17];
};

// Constant and non-constant offsetof expressions
void test_ice(int i) {
  int array0[__builtin_offsetof(HasArray, array[5])];
  int array1[__builtin_offsetof(HasArray, array[i])]; // expected-error{{variable length arrays are not permitted in C++}}
}

// Bitfields
struct has_bitfields {
  int i : 7;
  int j : 12; // expected-note{{bit-field is declared here}}
};

int test3 = __builtin_offsetof(struct has_bitfields, j); // expected-error{{cannot compute offset of bit-field 'j'}}
